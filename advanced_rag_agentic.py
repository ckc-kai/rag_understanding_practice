from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
import logging
import json
import os
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)
from ragas.llms import llm_factory
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import fitz

from config import setup_logger
from utils import get_chapter_nodes, ragas_evaluate

setup_logger()
logger = logging.getLogger(__name__)

logger.info("Setting up LLM and Embedding models...")
llm = Ollama(model="llama3.1", request_timeout=1200.0, context_window=8192, temperature=0.1)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")

logger.info("Loading chapter nodes...")
pdf_path = "./files/BuildTrap/BuildTrap.pdf"
nodes = get_chapter_nodes(pdf_path, lower_levels=0, upper_levels=1)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)
logger.info("Building Vector Operations Index...")
index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)
query_engine = index.as_query_engine(similarity_top_k=4, llm=llm)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="BuildTrap_RAG_Tool",
            description=(
                "Useful for answering questions about the book 'Escaping the Build Trap', "
                "product management, strategy, bad archetypes, and value exchange,"
                "anything related to the product management or development process."
                "If you find the answer in the tools, rely on the tool output to form your answer."
                "If you cannot find the answer in the tools, tell the user that you cannot find the answer and imply so."
            ),
        ),
    ),
    # QueryEngineTool(
    #     query_engine=query_engine,
    #     metadata=ToolMetadata(
    #         name="AgentDesign_RAG_Tool",
    #         description=(
    #             "Useful for answering questions about the book 'Agentic Design Patterns', "
    #             "a practical guide for developers on creating sophisticated AI agents"
    #             "If you find the answer in the tools, rely on the tool output to form your answer."
    #             "If you cannot find the answer in the tools, tell the user that you cannot find the answer and imply so."
    #         ),
    #     ),
    # )
]


agent = ReActAgent(
    tools=tools,
    llm=llm,
    system_prompt="You are a librarian agent. Select the correct book tool to answer the user's question.",
    verbose=True,
    name="SummarizerAgent",
    description="Agent that answers questions using books",
)

logger.info("Starting evaluation loop...")

questions_path = './files/BuildTrap/eval_questions.json'
if not os.path.exists(questions_path):
    logger.error(f"Questions file not found at {questions_path}")
    # Continue anyway to show structure or crash expectedly, but better to exit if real run
    exit(1)

with open(questions_path, 'r') as f:
    questions = json.load(f)

import asyncio

async def main():
    records = []
    output_file = './files/BuildTrap/generated_answer/advanced_agentic.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Helper for async execution
    async def run_agent_query(q):
        handler = agent.run(user_msg=q)
        return await handler

    for i, q in enumerate(questions):
        question = q['question']
        ground_truth = q['answer']
        
        logger.info(f"Processing Query {i+1}/{len(questions)}: {question}")
        
        try:
            # Await directly since we are inside an async function
            response = await run_agent_query(question)
            
            answer_text = str(response)
            
            # Extract context
            contexts = []
            if hasattr(response, 'source_nodes'):
                contexts = [node.node.get_text() for node in response.source_nodes]
                    
            if not contexts:
                contexts = ["No context retrieved."]
        
            records.append({
                    "question": question,
                    "answer": answer_text,
                    "contexts": contexts,
                    "ground_truth": ground_truth
                })
            logger.info(f"Response: {answer_text[:100]}...")
        
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            continue

    # Save generation results
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)

    # RAGAS Evaluation
    result = ragas_evaluate(records)
    logger.info(f"RAGAS Results: {result}")

if __name__ == "__main__":
    asyncio.run(main())
