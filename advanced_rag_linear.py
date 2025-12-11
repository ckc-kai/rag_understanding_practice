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

# # Custom Prompt Template
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "Don't use your own knowledge, only use the information provided in the context."
)
qa_template = PromptTemplate(template)

logger.info("Query Engine...")
# Query Engine and reranking
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=10,
    node_postprocessors=[reranker],
    text_qa_template=qa_template
)

with open('./files/eval_questions.json', 'r') as f:
    questions = json.load(f)

records = []
for q in questions:
    question = q['question']
    answer = q['answer']

    response = query_engine.query(question) 
    logger.info(f"Response: {response}")

    records.append({
        "question": question,
        "response": str(response),
        "contexts": [n.text for n in response.source_nodes], # the context window is within source_nodes
        "ground_truth": answer
    })

with open('./files/generated_answer/advanced_linear_1.json', 'w') as f:
    json.dump(records, f)

# Ragas evaluation 
ragas_embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

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
    metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
    llm=rag_llm,
    embeddings=ragas_embed_model,
    run_config=RunConfig(timeout=600, max_workers=1)
)

logger.info(f"Ragas results: {results}")