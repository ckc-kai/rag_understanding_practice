from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
import logging
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, Document, VectorStoreIndex, PromptTemplate
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
import json
from utils import ragas_evaluate, answer_questions
from configs.config import setup_logger
import os
logger = logging.getLogger(__name__)
from datetime import datetime

llm = Ollama(model="deepseek-r1:7b", request_timeout=1200.0, context_window=8192)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

class linear_rag_baseline():
    '''
    This class implements a linear basic RAG pipeline that includes the following steps:
    1. Load documents
    2. Split documents into chunks
    3. Create index
    4. Rerank
    5. A prompt template to force the LLM to use the context
    6. Setup query engine
    '''
    def __init__(self, document_path, llm, embed_model, chunk_sizes=[1024, 512]):
        self.doc_path = document_path
        self.llm = llm
        self.embed_model = embed_model
        self.chunk_sizes = chunk_sizes
        self.query_engine = None
        
    def setup_rag(self):
        logger.info("Loading documents...")
        reader = SimpleDirectoryReader(input_files=[self.doc_path])
        documents = reader.load_data()
        
        logger.info("Splitting documents into chunks...")
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        
        index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            llm=self.llm
        )
        
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=7,
            device="cpu"
        )
        
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
            "Don't use your own knowledge, only use the information provided in the context."
        )
        qa_template = PromptTemplate(template)
        
        self.query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=7,
            node_postprocessors=[reranker],
            text_qa_template=qa_template
        )
        logger.info("RAG pipeline setup complete.")
        
    def query(self, question):
        if not self.query_engine:
            raise ValueError("Run setup_rag() first!")
        return self.query_engine.query(question)

    def answer(self, questions_path, output_path):
        logger.info("Running query engine to answer questions...")
        eval_records = answer_questions(questions_path, output_path, self.query_engine)
        return eval_records

    def evaluate(self, records):
        logger.info("Running Ragas Evaluation...")
        evaluate_results = ragas_evaluate(records)
        return evaluate_results
        