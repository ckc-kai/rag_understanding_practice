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
        
        leaf_nodes = get_chapter_nodes(self.document_path, 1, 4)
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
            model="BAAI/bge-reranker-base",
            top_n=3,
            device="cpu"
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
# if __name__ == "__main__":
#     # Objectives: include HyDE, BM2

#     llm = Ollama(model="deepseek-r1:7b", request_timeout=1200.0, context_window=8192)
#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

#     # logger.info("Loading documents...")
#     # documents = SimpleDirectoryReader(
#     #     input_files = ["./files/BuildTrap.pdf"]
#     # ).load_data()

#     logger.info("Splitting documents into chunks for embedding...")

#     ## Split documents into chunks for embedding
#     # node_parser = HierarchicalNodeParser.from_defaults(
#     #     chunk_sizes=[1024, 512]
#     # )
#     # nodes = node_parser.get_nodes_from_documents(documents)
#     # leaf_nodes = get_leaf_nodes(nodes)  # only embed leaf nodes

#     # Use Chapter Chunking
#     nodes = get_chapter_nodes("./files/BuildTrap.pdf")
#     leaf_nodes = nodes # For chapter based chunking, all nodes are leaf nodes (flat structure)

#     # Link parents and children
#     # Storage context stores not only documents(text) but also nodes(embeddings)
#     storage_context = StorageContext.from_defaults()
#     storage_context.docstore.add_documents(nodes)

#     # Search logic 
#     # We use the leaf_nodes specifically for the index (embedding search), 
#     # while the storage_context holds the full tree (parents + children).
#     # Bge-small for Dense Search
#     vector_index = VectorStoreIndex(
#         leaf_nodes,
#         storage_context=storage_context,
#         embed_model=embed_model, 
#         llm=llm
#     )

#     logger.info("Building BM25 Index...")
#     # We use the same nodes for BM25 (Sparse Search)
#     bm25_retriever = BM25Retriever.from_defaults(
#         nodes=leaf_nodes,
#         similarity_top_k=10,
#         stemmer=Stemmer.Stemmer("english"),
#         language="english"
#     )

#     logger.info("Setting up Hybrid Fusion...")
#     vector_retriever = vector_index.as_retriever(similarity_top_k=10)

#     # Fusion Retriever combines BM25 + Vector results
#     fusion_retriever = QueryFusionRetriever(
#         [vector_retriever, bm25_retriever],
#         similarity_top_k=7, # Resulting top K for generation
#         num_queries=1,      # No multi-query generation here, we rely on HyDE
#         mode="reciprocal_rerank", # RRF for fusion
#         use_async=False,
#         verbose=True,
#         llm=llm
#     )

#     logger.info("Reranking...")
#     # Rerank the Fused Results (Optional but recommended)
#     reranker = SentenceTransformerRerank(
#         model="BAAI/bge-reranker-base",
#         top_n=5,
#         device="cpu"
#     )

#     #Hypothetical Query Transform
#     logger.info("Setting up HyDE...")
#     hyde = HyDEQueryTransform(include_original=True, llm=llm)


#     logger.info("Assembling Query Engine...")

#     # Base engine with fusion retriever
#     base_query_engine = RetrieverQueryEngine.from_args(
#         retriever=fusion_retriever,
#         llm=llm,
#         node_postprocessors=[reranker]
#     )

#     # Wrap with HyDE
#     # Note: TransformQueryEngine applies the transform (HyDE) to the query BEFORE passing it to base_query_engine
#     hyde_query_engine = TransformQueryEngine(base_query_engine, query_transform=hyde)


#     # Step Decompose Query
#     # synthesizer = get_response_synthesizer(llm=llm)
#     # step_decompose_transform = StepDecomposeQueryTransform(llm, verbose=True)
#     # step_decompose_query_engine = MultiStepQueryEngine(base_query_engine, query_transform=step_decompose_transform, response_synthesizer=synthesizer)

#     with open('./files/eval_questions.json', 'r') as f:
#         questions = json.load(f)

#     records = []
#     for q in questions:
#         question = q['question']
#         answer = q['answer']
        
#         logger.info(f"Querying: {question}")
#         response = hyde_query_engine.query(question) 
#         #response = step_decompose_query_engine.query(question)
#         logger.info(f"Response: {response}")

#         records.append({
#             "question": question,
#             "response": str(response),
#             "contexts": [n.text for n in response.source_nodes],
#             "ground_truth": answer
#         })

#     with open('./files/generated_answer/advanced_linear_2.json', 'a') as f:
#         json.dump(records, f)

#     # Ragas evaluation 
#     ragas_embed_model = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-small-en-v1.5",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     client = OpenAI(
#         base_url="http://localhost:11434/v1",
#         api_key="ollama"
#     )
#     rag_llm = llm_factory(
#         model="qwen2.5:7b-instruct", 
#         client=client,
#         provider="openai"
#     )

#     dataset = Dataset.from_list(records)

#     logger.info("Ragas evaluation...")
#     results = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
#         llm=rag_llm,
#         embeddings=ragas_embed_model,
#         run_config=RunConfig(timeout=600, max_workers=1)
#     )

#     logger.info(f"Ragas results: {results}")
