from config import setup_logger
import os
import logging 
from advanced_rag_linear import linear_rag_baseline
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Runs everything here in the main

TEST_DOCUMNET_PATH = "./files/AgenticDesign/Agentic_Design_Pattern.pdf"
TEST_QUESTIONS_PATH = "./files/AgenticDesign/eval_questions.json"
TEST_OUTPUT_PATH = "./files/AgenticDesign/generated_answer/linear_1.json"

def model_setup(llm, embed_model):
    logger.info("Setting up model...")
    llm = Ollama(model=llm, request_timeout=1200.0, context_window=8192)
    embed_model = HuggingFaceEmbedding(model_name=embed_model, device="cpu")
    return llm, embed_model

if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(TEST_DOCUMNET_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_QUESTIONS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_OUTPUT_PATH), exist_ok=True)
    llm, embed_model = model_setup("deepseek-r1:7b", "BAAI/bge-small-en-v1.5")

    simple_rag = linear_rag_baseline(
        document_path=TEST_DOCUMNET_PATH,
        llm=llm,
        embed_model=embed_model
    )
    simple_rag.setup_rag()

    records = simple_rag.answer(
        questions_path=TEST_QUESTIONS_PATH,
        output_path=TEST_OUTPUT_PATH
    )

    # Ragas evaluation
    results = simple_rag.evaluate(
        records=records,
        output_path="./files/AgenticDesign/eval_results/linear_1.json"
    )
    logger.info(f"Ragas results: {results}")

