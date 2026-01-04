from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
import logging
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, Document, VectorStoreIndex, PromptTemplate, get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, StepDecomposeQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine, MultiStepQueryEngine

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
import Stemmer 
import fitz
from utils import get_chapter_nodes, answer_questions, ragas_evaluate
from configs.config import setup_logger

logger = logging.getLogger(__name__)

class linear_rag_chapter_based():
    '''
    This workflow is based on the linear base rag.
    Some advanced features include:
    1. chunk based on chapter content
    2. Hybrid Search (BM25 + Vector)
    3. HyDE Query Transform
    '''
    def __init__(self, llm, embed_model, document_path):
        self.llm = llm
        self.embed_model = embed_model
        self.document_path = document_path
        self.query_engine = None 

    def setup_rag(self):
        logger.info("Start to setup the rag...")
        
        leaf_nodes = get_chapter_nodes(self.document_path, 2)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(leaf_nodes)

        vector_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            llm=self.llm
        )
        logger.info("Sparse Search with BM25...")
        bm25_retriever = BM25Retriever.from_defaults(
            nodes = leaf_nodes,
            similarity_top_k=9,
            stemmer=Stemmer.Stemmer("english"),
            language="english"
        )
        vector_retriever = vector_index.as_retriever(similarity_top_k=7)
        logger.info("Hybrid Fusion Search with Sparse and Dense Search...")
        fusion_retriever = QueryFusionRetriever(
            [bm25_retriever, vector_retriever],
            similarity_top_k=5,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
            llm=self.llm
        )
        
        logger.info("Reranking...")
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-large",
            top_n=3,
            device="mps"
        )

        logger.info("Setup HyDE...")
        hyde = HyDEQueryTransform(include_original=True, llm=self.llm)

        logger.info("Assembling Query Engine...")
        base_query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            llm=self.llm,
            node_postprocessors=[reranker]
        )

        hyde_query_engine = TransformQueryEngine(base_query_engine, query_transform=hyde)

        self.query_engine = hyde_query_engine

    def query(self, query):
        if not self.query_engine:
            raise ValueError("Query engine is not initialized. Please call setup_rag() first.")
        return self.query_engine.query(query)

    def answer(self, question_path, out_path):
        logger.info("Running query engine to answer loaded questions...")
        evaluate_records = answer_questions(question_path, out_path, self.query_engine)
        return evaluate_records 
    def evaluate(self, records):
        logger.info("Running Ragas Evaluation...")
        evaluate_results = ragas_evaluate(records)
        return evaluate_results 
