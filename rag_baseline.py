import torch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Document
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from datasets import Dataset
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Provide the documents
documents = SimpleDirectoryReader(
    input_files = ["./files/Kaicheng_Chu_Resume copy.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

# Basic RAG
# Models
llm = Ollama(model="deepseek-r1:7b", request_timeout=1200.0, context_window=8192)
# Embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

# Index
index = VectorStoreIndex.from_documents([document], llm=llm, embed_model=embed_model)

query_engine = index.as_query_engine(llm = llm, similarity_top_k=3)

# Questions
questions = [
    "What is the most matching point of this applicant to the DS Intern position?",
    "What is the weakest point of this applicant?",
    "How can the applicant improve further?"
]

records = []
for q in questions:
    response = query_engine.query(q)
    records.append({
        "question": q,
        "response": str(response),
        "contexts": [n.text for n in response.source_nodes],
        "ground_truth": ""
    })

from openai import OpenAI

# ... (imports)

# RAGAS Evaluation
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

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=rag_llm,
)

print(results)