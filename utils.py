import logging
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
import fitz
from configs.config import setup_logger
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
import os
import json
from datetime import datetime
logger = logging.getLogger(__name__)

def get_chapter_nodes(pdf_path, upper_levels=9999):
    '''
    This is a function that extract the level 1 title and page number from the pdf file.
    '''
    doc = fitz.open(pdf_path)
    # Get toc
    toc = doc.get_toc(simple=True)
    
    if not toc:
        logger.warning("No TOC found! Falling back to full text.")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return [TextNode(text=full_text, metadata={"title": "Full Document"})]

    # Convert to 0-based indices and prepare list
    chapters = []
    for entry in toc:
        level, title, page_num = entry
        if level <= upper_levels:
            start_idx = page_num - 1 
            chapters.append({
                "title": title,
                "start_idx": start_idx
            })

    chapters.sort(key=lambda x: x['start_idx'])
    
    nodes = []
    for i, chapter in enumerate(chapters):
        start_idx = chapter['start_idx']
        
        # End page is the start of next chapter, or last page
        if i < len(chapters) - 1:
            end_idx = chapters[i+1]['start_idx']
        else:
            end_idx = len(doc)
            
        # Extract text
        chapter_text = ""
        # fitz pages are 0-indexed
        for p in range(start_idx, end_idx):
            chapter_text += doc[p].get_text() + "\n"
            
        # Sub-chunk the chapter text to fit embedding model (512 tokens)
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(chapter_text)
        
        for chunk_text in chunks:
            node = TextNode(
                text=chunk_text,
                metadata={
                    "title": chapter['title'],
                    "start_page_idx": start_idx,
                    "end_page_idx": end_idx
                }
            )
            nodes.append(node)
        
    logger.info(f"Created {len(nodes)} nodes from chapters using PyMuPDF (chunked to 512 tokens).")
    return nodes

def ragas_evaluate(records):
    '''
    This function is used to evaluate the generated answers using Ragas.
    The embedding model uses BAAI/bge-small-en-v1.5 and the LLM uses Qwen2.5:7b-instruct.
    Args:
        records (list): List of records to evaluate.
    Returns:
        dict: Dictionary with the following keys:
        - faithfulness
        - answer_relevancy
        - context_precision
        - answer_correctness
    '''
    logger.info("Starting Ragas evaluation...")

    ragas_embedding = HuggingFaceEmbeddings(
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
        
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
        llm=rag_llm,
        embeddings=ragas_embedding,
        run_config=RunConfig(timeout=600, max_workers=1)
    )
    return results

def answer_questions(question_path, output_path, query_engine):
    '''
        This function is used to evaluate the generated answers using Ragas.
        The embedding model uses BAAI/bge-small-en-v1.5 and the LLM uses Qwen2.5:7b-instruct.
        return format is a dictionary with the following keys:
        - faithfulness
        - answer_relevancy
        - context_precision
        - answer_correctness
        return format is a list of dictionaries with the following keys:
        - user_input
        - response
        - retrieved_contexts
        - reference
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(question_path), exist_ok=True)

    # Load questions
    with open(question_path, 'r') as f:
        questions = json.load(f)
        
    if len(questions) == 0:
        raise ValueError("No questions found in the question file.")
    if not query_engine:
        raise ValueError("Query engine not initialized.")
        
    eval_records = []
    for q in questions:
        question = q['question']
        answer = q['answer']
            
        response = query_engine.query(question)
            
        logger.info(f"Answered: {question}")
        eval_records.append({
            "user_input": question,
            "response": str(response),
            "retrieved_contexts": [n.text for n in response.source_nodes],
            "reference": answer,
        })
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_records = [{"timestamp": timestamp}] + eval_records

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = []
    if os.path.exists(output_path):
        with open(output_path,"r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
                    
    if isinstance(data, list):
        data.append(file_records) 
    else:
        data = [file_records]

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return eval_records

