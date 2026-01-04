from configs.config import setup_logger
import os
import logging 
from advanced_rag_linear import linear_rag_baseline
from advanced_rag_linear2 import linear_rag_chapter_based
from llama_index.llms.ollama import Ollama
import json
from utils import ragas_evaluate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Runs everything here in the main

TEST_DOCUMNET_PATH = "./files/AgenticDesign/Agentic_Design_Pattern.pdf"
TEST_QUESTIONS_PATH = "./files/AgenticDesign/eval_questions.json"
TEST_OUTPUT_PATH = "./files/AgenticDesign/generated_answer/linear_2.json"

def model_setup(llm, embed_model):
    logger.info("Setting up model...")
    llm = Ollama(model=llm, request_timeout=1200.0, context_window=8192)
    embed_model = HuggingFaceEmbedding(model_name=embed_model, device="mps")
    return llm, embed_model

if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(TEST_DOCUMNET_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_QUESTIONS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_OUTPUT_PATH), exist_ok=True)
    # llm, embed_model = model_setup("qwen2.5:14b-instruct", "BAAI/bge-large-en-v1.5")

    # simple_rag = linear_rag_baseline(
    #     document_path=TEST_DOCUMNET_PATH,
    #     llm=llm,
    #     embed_model=embed_model
    # )
    # simple_rag.setup_rag()

    # chapter_rag = linear_rag_chapter_based(
    #     document_path=TEST_DOCUMNET_PATH,
    #     llm=llm,
    #     embed_model=embed_model
    # )
    # chapter_rag.setup_rag()

    # records = chapter_rag.answer(
    #     question_path=TEST_QUESTIONS_PATH,
    #     out_path=TEST_OUTPUT_PATH
    # )

    # # Ragas evaluation
    # results = chapter_rag.evaluate(
    #     records=records,
    # )
    # logger.info(f"Ragas results: {results}")
    with open(TEST_OUTPUT_PATH, "r") as f:
        generated_answer = json.load(f)
    
    # Process the data to match Ragas requirements
    ragas_records = []
    for item in generated_answer:
        # item is a list where the second element is the actual record
        if isinstance(item, list) and len(item) > 1:
            record = item[1]
            ragas_record = {
                "question": record.get("user_input"),
                "answer": record.get("response"),
                "contexts": record.get("retrieved_contexts"),
                "ground_truth": record.get("reference")
            }
            ragas_records.append(ragas_record)
            
    results = ragas_evaluate(ragas_records)
    logger.info(f"Ragas results: {results}")

