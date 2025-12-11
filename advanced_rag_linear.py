from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
import logging
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, Document, VectorStoreIndex
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
import json

from config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

llm = Ollama(model="deepseek-r1:7b", request_timeout=1200.0, context_window=8192)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

logger.info("Loading documents...")
documents = SimpleDirectoryReader(
    input_files = ["./files/BuildTrap.pdf"]
).load_data()

logger.info("Splitting documents into chunks for embedding...")
# Split documents into chunks for embedding
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)  # only embed leaf nodes

# Link parents and children
# Storage context stores not only documents(text) but also nodes(embeddings)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# Search logic 
# We use the leaf_nodes specifically for the index (embedding search), 
# while the storage_context holds the full tree (parents + children).
index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
    embed_model=embed_model, 
    llm=llm
)

logger.info("Reranking...")
# Pick top 3 nodes
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
    device="cpu"
)

logger.info("Query Engine...")
# Query Engine and reranking
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=10,
    node_postprocessors=[reranker]
)

with open('./files/eval_questions.json', 'r') as f:
    questions = json.load(f)

records = []
# Test question 1
question = questions[0]['question']
answer = questions[0]['answer']

response = query_engine.query(question) 
logger.info(f"Response: {response}")

records.append({
    "question": question,
    "response": str(response),
    "contexts": [n.text for n in response.source_nodes], # the context window is within source_nodes
    "ground_truth": answer
})

logger.info(f"Records: {records}")

# Ragas evaluation 
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
rag_llm = llm_factory(
    model="qwen2.5:7b-instruct", 
    client=client,
    provider="openai"
)

dataset = Dataset.from_list(records)

logger.info("Ragas evaluation...")
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=rag_llm,
)

logger.info(f"Ragas results: {results}")