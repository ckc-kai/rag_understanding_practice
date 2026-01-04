from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
import Stemmer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
import logging
import json
import os
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness
)
from ragas.llms import llm_factory
from openai import OpenAI
import fitz
from utils import get_chapter_nodes, ragas_evaluate
from collections import defaultdict
from configs.config import setup_logger
logger = logging.getLogger(__name__)
from dataclasses import dataclass, asdict
import numpy as np
import asyncio


@dataclass
class RetrievalConfig:
    use_reranking: bool
    rerank_model: str
    use_query_expansion: bool
    similarity_top_k: int
    rerank_top_n: int
    similarity_cutoff: float

@dataclass
class BookConfig:
    name: str
    pdf_path: str
    description: str
    keywords: List[str]
    category: str = "general"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5   
    rerank_top_n: int = 3
    similarity_cutoff: float = 0.7
    enabled: bool = True
@dataclass
class SystemConfig:
    """Complete system configuration"""
    books: List[BookConfig]
    llm_config: Dict[str, Any]
    embed_config: Dict[str, Any]
    agent_config: Dict[str, Any]
    retrieval_config: RetrievalConfig
    cache_config: Dict[str, Any]

class ConfigLoader:
    '''
    Load and validate configuaration from YAML file
    '''
    @staticmethod
    def load(book_config_path: str = "./configs/books_config.yaml", model_config_path: str = "./configs/models_config.yaml") -> SystemConfig:
        logger.info("Loading configuration...")

        if not os.path.exists(book_config_path):
            raise FileNotFoundError(f"Book config file not found at {book_config_path}")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found at {model_config_path}")

        with open(book_config_path, "r") as f:
            book_config = yaml.safe_load(f)
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        books = []
        for book in book_config.get('books', []):
            if book.get('enabled', True):
                books.append(BookConfig(**book))
        retrieval_cfg = model_config.get("retrievals")
        return SystemConfig(
            books=books,
            llm_config=model_config['models']['llm'],
            embed_config=model_config['models']['embedding'],
            agent_config=model_config['agent'],
            retrieval_config=RetrievalConfig(**retrieval_cfg),
            cache_config=model_config.get('cache', {}),
        )

class QueryDecomposer:
    '''
    Decompose query into subqueries
    '''
    def __init__(self, llm):
        self.llm = llm
    def decompose(self, query: str) -> List[str]:
        prompt = f'''
            Given this complex question, break it down into a list of subqueries
            that can be answered by a single chunk of text. Each sub-question should be answerable independently.
            
            Complex Question: {query}
            
            Return ONLY a JSON array of sub-questions, nothing else.
            Example: ["What is X?", "How does Y work?", "What's the relationship between X and Y?"]
            
            Sub-questions:
        '''
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()
            if '[' in response_text:
                start = response_text.index('[')
                end = response_text.rindex(']') + 1
                json_str = response_text[start:end]
                sub_queries = json.loads(json_str)
                
                logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
                return sub_queries
            else:
                return [query]  # Fallback
        except Exception as e:
            logger.warning(f"Failed to decompose query: {e}")
            return [query]

class AnswerValidator:
    '''
    Validate answer
    '''
    def __init__(self, llm):
        self.llm = llm
    
    def validate(self, question: str, answer: str, context: List[str]) -> Dict[str, Any]:
        prompt = f"""Evaluate if this answer adequately addresses the question.

Question: {question}

Answer: {answer}

Evaluation criteria:
1. Completeness: Does it fully answer the question?
2. Accuracy: Is the information correct based on context?
3. Clarity: Is it well-explained?
4. Relevance: Does it stay on topic?

Return a JSON object with:
{{
  "is_adequate": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"]
}}

Evaluation:"""
        
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            if '{' in response_text:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                json_str = response_text[start:end]
                # Fix common boolean issues just in case
                json_str = json_str.replace("True", "true").replace("False", "false")
                evaluation = json.loads(json_str)
                
                logger.info(f"Evaluation: {evaluation}")
                return evaluation
            else:
                return {"is_adequate": False, "confidence": 0.0, "issues": ["Invalid JSON format"], "suggestions": ["Check the answer format"]}  # Fallback
        except Exception as e:
            logger.warning(f"Failed to validate answer: {e}")
            logger.warning(f"Raw LLM response causing failure: {response_text}")
            return {"is_adequate": False, "confidence": 0.0, "issues": ["Evaluation failed"], "suggestions": ["Check the answer format"]}

class CrossBookSynthesizer:
    '''
    Synthesize answer from multiple books
    '''
    def __init__(self, llm):
        self.llm = llm
    
    def synthesize(self, question: str, book_answers: Dict[str, str]) -> str:
        '''
        Synthesize answer from multiple books
        '''
        books_info = "\n\n".join([
            f"Book: {book}\nAnswer: {answer}"
            for book, answer in book_answers.items()
        ])
        prompt = f"""You are synthesizing information from multiple books to answer a question.

Question: {question}

Information from different books:
{books_info}

Task: Create a comprehensive answer that:
1. Integrates insights from all sources
2. Highlights agreements and disagreements between sources
3. Provides a balanced perspective
4. Cites which book each piece of information comes from
5. Build the bridge to connect the dots between different books

Synthesized Answer:"""
        
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()
            return response_text
        except Exception as e:
            logger.warning(f"Failed to synthesize answer: {e}")
            return "Failed to synthesize answer"

class AgenticMemory:
    """
    Agent memory to store conversation history and context using embeddings
    """
    def __init__(self, embed_model, json_path: str = "./cache/memory/agent_memory.json"):
        self.embed_model = embed_model
        self.json_path = json_path
        self.query_history = []
        self.book_usage = defaultdict(int)
        self.success_patterns = []
        self.load()

    def load(self):
        """Load memory from JSON file"""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.query_history = data.get("query_history", [])
                    # Reconstruct book usage
                    for item in self.query_history:
                        if item.get("success"):
                            for book in item.get("books_used", []):
                                self.book_usage[book] += 1
                logger.info(f"Loaded {len(self.query_history)} items from memory")
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")

    def save(self):
        """Save memory to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            data = {
                "query_history": self.query_history
            }
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved memory to disk")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        try:
            return self.embed_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []

    def add_query(self, question: str, answer: str, books_used: List[str], success: bool):
        '''
        Add a query to the history
        '''
        embedding = self._get_embedding(question)
        self.query_history.append({
            "question": question,
            "embedding": embedding,
            "answer": answer,
            "books_used": books_used,
            "success": success
        })
        if success:
            for book in books_used:
                self.book_usage[book] += 1
        self.save()
                
    def get_relevant_history(self, current_question: str, n: int = 3) -> List[Dict]:
        '''Retrieve similar past queries using cosine similarity'''
        if not self.query_history:
            return []

        current_embedding = self._get_embedding(current_question)
        if not current_embedding:
            return []
            
        scored_history = []
        for item in self.query_history:
            if not item.get("embedding"):
                continue
            
            # Simple cosine similarity
            vec1 = np.array(current_embedding)
            vec2 = np.array(item["embedding"])
            
            # handle zero vectors
            if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                score = 0
            else:
                score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
            scored_history.append((score, item))
        
        scored_history.sort(reverse=True, key=lambda x: x[0])
        # Return top N items (stripping the score for cleaner output if needed, or keeping it)
        return [item for score, item in scored_history[:n]]
    
    def suggest_books(self, question: str) -> List[str]:
        """
        Suggest books based on successful patterns
        """
        relevant = self.get_relevant_history(question, n=5)
        book_scores = defaultdict(int)
        
        for item in relevant:
            if item["success"]:
                for book in item["books_used"]:
                    book_scores[book] += 1
        
        return sorted(book_scores.keys(), key=lambda b: book_scores[b], reverse=True)

class QueryPlanner:
    """
    Plans to execute queries
    """
    def __init__(self, llm, available_books: List[BookConfig]):
        self.llm = llm
        self.available_books = available_books
    
    def create_plan(self, question: str, memory: Optional[AgenticMemory] = None) -> Dict[str, Any]:
        suggested_books = []
        if memory:
            # Get all suggestions without slicing
            suggested_books = memory.suggest_books(question)
            logger.info(f"Memory suggested books: {suggested_books}")
            
        # Format detailed book info for the prompt
        books_info = "\n".join([
            f"- Name: {b.name}\n  Description: {b.description}\n  Keywords: {', '.join(b.keywords)}"
            for b in self.available_books
        ])
        
        prompt = f"""Create an execution plan to answer this question.
You have access to the following books. You must be CRITICAL in selecting books. 
Ask yourself: "Do I really need this book? Does the question's intent match the book's description and keywords?"
Only select a book if it is strictly necessary.

Available Books:
{books_info}

Previously successful books for similar questions: {', '.join(suggested_books) if suggested_books else "None"}

Question: {question}

Tasks:
1. Analyze the question's core intent.
2. Determine if the question contains multiple distinct sub-questions (regardless of complexity).
   - "dependent": Question parts must be answered in order (e.g., "Find X, then use X to do Y").
   - "independent": Question parts can be answered separately (e.g., "What is A and what is B?").
   - "none": The question is a single, atomic unit.
   *CRITICAL*: Even if a question is simple (e.g., "Define A and B"), if it asks for two distinct things, it NEEDS decomposition into Independent sub-queries.
3. Select ONLY the books that are relevant based on their name, description, and keywords. Do not select a book just because it is available.
4. If the question involves multiple books, select all books that are relevant to the question.

Return ONLY a JSON object:
{{
  "complexity": "simple|moderate|complex",
  "needs_decomposition": true/false,
  "decomposition_mode": "dependent|independent|none",
  "books_to_query": ["book(s)"],
  "needs_synthesis": true/false,
  "reasoning": "Detailed reasoning for book selection and decomposition"
}}

Plan:"""
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()
            if '{' in response_text:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                json_str = response_text[start:end]
                json_str = json_str.replace("True", "true").replace("False", "false")
                plan = json.loads(json_str)
                logger.info(f"Created plan: {plan.get('complexity')} query")
                return plan
            else:
                logger.warning(f"No JSON found in response: {response_text}")
                return {
                    "complexity": "simple",
                    "needs_decomposition": False,
                    "books_to_query": [self.available_books[0].name] if self.available_books else [],
                    "needs_synthesis": False,
                    "reasoning": "Fallback plan"
                }
        except Exception as e:
            logger.warning(f"Planning failed: {e}")
            logger.warning(f"Raw LLM response causing failure: {response_text}")
            return {
                "complexity": "simple",
                "needs_decomposition": False,
                "books_to_query": [self.available_books[0].name] if self.available_books else [],
                "needs_synthesis": False
            }

# Orchestrator
class AgentOrchestrator:
    def __init__(self, agent: ReActAgent, llm, embed_model, book_configs:List[BookConfig], tools: List[QueryEngineTool]):
        self.agent = agent
        self.llm = llm
        self.embed_model = embed_model
        self.book_configs = book_configs
        self.tools = {tool.metadata.name: tool for tool in tools}
        
        self.decomposer = QueryDecomposer(llm)
        self.validator = AnswerValidator(llm)
        self.synthesizer = CrossBookSynthesizer(llm)
        self.long_term_memory = AgenticMemory(embed_model)
        self.planner = QueryPlanner(llm, book_configs)

        self.stats = {
            "total_queries": 0,
            "decomposed_queries": 0,
            "multi_book_queries": 0,
            "retries": 0,
            "syntheses": 0
        }
    async def _execute_sub_queries(self, sub_queries: List[str], plan: Dict[str, Any]) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Execute sub-queries based on dependency mode (dependent vs independent).

        Args:
            sub_queries (List[str]): List of sub-queries to execute.
            plan (Dict[str, Any]): Execution plan containing decomposition mode.
        Returns:
            Tuple[Dict[str, str], List[str], List[str]]: Tuple containing book answers, contexts, and used books.
        """
        mode = plan.get("decomposition_mode", "independent")
        all_book_answers = {}
        all_contexts = []
        all_books_used = set()
        
        if mode == "dependent":
            logger.info("[EXECUTION] Running DEPENDENT sub-queries...")
            context_accumulator = ""
            for i, sq in enumerate(sub_queries):
                # Augment subsequent queries with previous context
                current_query = sq
                if context_accumulator:
                    current_query = f"{sq}\n\nContext from previous steps:\n{context_accumulator}"
                
                logger.info(f"[STEP {i+1}] Querying: {sq}")
                step_answers, step_contexts, step_books = await self._execute_single_step(current_query, plan)
                
                # Accumulate answers
                step_combined_answer = "\n".join(step_answers.values())
                context_accumulator += f"\n-- Step {i+1} Answer --\n{step_combined_answer}"
                
                # Merge answers
                all_book_answers.update(step_answers)
                all_contexts.extend(step_contexts)
                all_books_used.update(step_books)
                
        else:
            #TODO: This Independent sub-queries are still running sequentially.
            logger.info("[EXECUTION] Running INDEPENDENT sub-queries...")
            for i, sq in enumerate(sub_queries):
                logger.info(f"[STEP {i+1}] Querying: {sq}")
                step_answers, step_contexts, step_books = await self._execute_single_step(sq, plan)
                
                for book, ans in step_answers.items():
                    if book in all_book_answers:
                        all_book_answers[book] += f"\n\n[Part {i+1}]: {ans}"
                    else:
                        all_book_answers[book] = f"[Part {i+1}]: {ans}"
                all_contexts.extend(step_contexts)
                all_books_used.update(step_books)
                
        return all_book_answers, all_contexts, list(all_books_used)

    async def _execute_single_step(self, question: str, plan: Dict[str, Any]) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Execute a single query by performing explicit retrieval then agent synthesis.

        Args:
            question (str): The question to execute.
            plan (Dict[str, Any]): The execution plan.
        Returns:
            Tuple[Dict[str, str], List[str], List[str]]: Tuple containing book answers, contexts, and used books.
        """
        book_answers = {}
        contexts = []
        used_books = set()
        
        target_books = plan.get("books_to_query", [])
        if not target_books:
            logger.info("[EXECUTION] No books selected by planner. Returning early.")
            return {
                "agent_response": "No relevant books found in the library to answer this specific question based on the planner's assessment."
            }, [], [] 

        if len(target_books) > 1:
            logger.info(f"[MULTI-BOOK] Querying {len(target_books)} books: {target_books}")
            self.stats["multi_book_queries"] += 1
        
        # --- Step 1: Explicit Retrieval ---
        current_contexts = []
        for book_name in target_books:
            tool_name = f"{book_name}_Tool"
            tool = self.tools.get(tool_name)
            
            if not tool:
                logger.warning(f"Tool {tool_name} not found in orchestrator tools.")
                continue
                
            try:
                logger.info(f"[RETRIEVAL] Querying {book_name} directly...")
                # Direct query to the engine
                response = tool.query_engine.query(question)
                
                # Extract nodes
                if hasattr(response, "source_nodes"):
                    for node in response.source_nodes:
                        content = node.node.get_content()
                        # Add citation metadata to help the LLM reference correctly
                        current_contexts.append(f"Source: {book_name}\nContent: {content}")
                        used_books.add(book_name)
                
            except Exception as e:
                logger.error(f"[RETRIEVAL] Failed to query {book_name}: {e}")

        contexts.extend(current_contexts)
        
        # --- Step 2: Synthesis ---
        try:
            if current_contexts:
                context_str = "\n\n".join(current_contexts)
                prompt = f"""
You are an intelligent research assistant. 
Answer the following question using ONLY the provided context below.
Do not hallucinate information not present in the context.

Context:
{context_str}

Question: {question}

Answer:
"""
                logger.info(f"[EXECUTION] Synthesizing answer from {len(current_contexts)} chunks...")
                # Use standard completion/chat instead of agent.run because ReActAgent encounters Recursive Retreival Loops.
                response = await self.llm.acomplete(prompt) 
                book_answers["agent_response"] = str(response)
                
                logger.info(f"[EXECUTION] Success. Retrieved {len(contexts)} chunks from {len(used_books)} books.")
            
            else:
                logger.warning("[EXECUTION] No contexts retrieved. Asking agent to search.")
                response = await self.agent.run(question)
                book_answers["agent_response"] = str(response)

        except Exception as e:
            logger.error(f"[EXECUTION] Synthesis failed: {e}", exc_info=True)
            book_answers["error"] = str(e)
            
        return book_answers, contexts, list(used_books)

    async def query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        self.stats["total_queries"] += 1
        logger.info(f"AGENTIC QUERY: {question}")

        # 1. Create Execution Plan
        logger.info("[PLANNING] Creating execution plan...")
        plan = self.planner.create_plan(question, self.long_term_memory)
        logger.info(f"[PLANNING] Plan: {plan}")

        # 2. Decompose if needed
        sub_queries = [question]
        if bool(plan.get("needs_decomposition", False)):
            logger.info("[DECOMPOSITION] Decomposing query...")
            sub_queries = self.decomposer.decompose(question)
            self.stats["decomposed_queries"] += 1
        
        # 3. Execute logic (Dependent/Independent/Single)
        book_answers, all_contexts, books_used = await self._execute_sub_queries(sub_queries, plan)
        
        logger.info(f"[EXECUTION] All contexts: {all_contexts}")
        logger.info(f"[EXECUTION] Books used: {books_used}")

        # 4. Synthesize if needed (Already partly handled by accumulating answers, but we might want a final polish)
        if len(book_answers) > 1 and bool(plan.get("needs_synthesis", False)):
            logger.info("[SYNTHESIS] Combining resources...")
            try:
                final_answer = self.synthesizer.synthesize(question=question, book_answers=book_answers)
                self.stats["syntheses"] += 1
            except Exception as e:
                logger.error(f"[SYNTHESIS] Synthesis failed: {e}")
                final_answer = f"Error: {e}"
        else:
            final_answer = list(book_answers.values())[0] if book_answers else "No answer found"
        
        logger.info(f"Final answer: {final_answer}")
        
        # 5. Validate answer quality
        logger.info("[VALIDATION] Validating answer quality...")
        validation = self.validator.validate(question=question, answer=final_answer, context=all_contexts)
        
        # 6. Retry 
        retry_count = 0
        while (not bool(validation.get("is_adequate", True)) and retry_count < max_retries):
            if not all_contexts:
                logger.warning("Retry without context — skipping further retries.")
                break

            retry_count += 1
            logger.info(f"[RETRY] Retrying query ({retry_count}/{max_retries})...")
            self.stats["retries"] += 1
            
            issues = validation.get("issues", [])
            try:
                retry_prompt = f"""
You previously answered the question below, but the answer was judged inadequate.

Question:
{question}

Previous Answer:
{final_answer}

Issues Identified:
{chr(10).join(f"- {i}" for i in issues)}

Retrieved Context (grounding material):
{chr(10).join(all_contexts[:5])}

Task:
- Revise the previous answer to fix the issues.
- Preserve correct parts.
- Correct or remove incorrect or unsupported claims.
- Ground the answer strictly in the provided context.
- If context is insufficient, explicitly say so.

Return ONLY the revised answer.
"""
                response = await self.agent.run(retry_prompt)
                
                final_answer = str(response)
                logger.info(f"[RETRY] Validated New Answer: {final_answer}")

                validation = self.validator.validate(question=question, answer=final_answer, context=all_contexts)
            except Exception as e:
                logger.error(f"[RETRY] Retry failed: {e}")
                break
        
        # 7. Record to long-term memory
        # books_used is now accurately populated from execution steps
        
        success = bool(validation.get("is_adequate", True))
        self.long_term_memory.add_query(
            question=question,
            answer=final_answer,
            books_used=books_used,
            success=success
        )
                
        return {
            "question": question,
            "answer": final_answer,
            "plan": plan,
            "books_used": books_used,
            "sub_queries": sub_queries if len(sub_queries) > 1 else None,
            "validation": validation,
            "contexts": all_contexts,
            "retries": retry_count
        }
class Librarian:
    def __init__(self, book_config_path: str, model_config_path: str):
        self.config = ConfigLoader.load(book_config_path=book_config_path, model_config_path=model_config_path)
        self._initialize_models()
        self._build_library()
        self._create_agent()
        self._initialize_agentic_layer()
    def _initialize_models(self):
        logger.info("Initializing models...")
        self.llm = Ollama(**self.config.llm_config)
        
        # Prepare embedding config
        embed_config = self.config.embed_config.copy()
        device = embed_config.pop("device", "cpu")
        model_name = embed_config.pop("model_name", "BAAI/bge-large-en-v1.5")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            **embed_config
        )
    def _build_library(self):
        logger.info("Building library...")
        self.tools = []
        self.book_names = []
        
        # Ensure cache directory exists
        cache_dir = self.config.cache_config.get("cache_dir", "./cache/indexes")
        os.makedirs(cache_dir, exist_ok=True)

        for book in self.config.books:
            if not book.enabled:
                continue
            pdf_path = book.pdf_path
            if not os.path.exists(pdf_path):
                logger.warning(f"Book {book.name} not found at {pdf_path}")
                continue
            
            book_cache_path = os.path.join(cache_dir, book.name)
            
            try:
                # Try to load from cache first
                if os.path.exists(book_cache_path) and self.config.cache_config.get("enabled", False):
                    logger.info(f"Loading {book.name} index from cache...")
                    storage_context = StorageContext.from_defaults(persist_dir=book_cache_path)
                    vector_index = load_index_from_storage(storage_context, embed_model=self.embed_model)
                else:
                    logger.info(f"Building index for {book.name}...")
                    # Build index
                    nodes = get_chapter_nodes(pdf_path)
                    storage_context = StorageContext.from_defaults()
                    storage_context.docstore.add_documents(nodes)
                    vector_index = VectorStoreIndex(
                        nodes,
                        storage_context=storage_context,
                        embed_model=self.embed_model,
                        llm = self.llm
                    )
                    # Persist if caching is enabled
                    if self.config.cache_config.get("enabled", False):
                        logger.info(f"Persisting index for {book.name} to cache...")
                        vector_index.storage_context.persist(persist_dir=book_cache_path)

                # --- Hybrid Search Setup ---
                logger.info(f"Setting up Hybrid Search for {book.name}...")
                
                # BM25 Retriever (Sparse)
                # Note: We need nodes for BM25. If loaded from disk, we might need to reconstruct them or load docstore.
                if 'nodes' not in locals():
                     nodes = list(vector_index.docstore.docs.values())
                
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=self.config.retrieval_config.similarity_top_k * 2,
                    stemmer=Stemmer.Stemmer("english"),
                    language="english"
                )

                # Vector Retriever (Dense)
                vector_retriever = vector_index.as_retriever(
                    similarity_top_k=self.config.retrieval_config.similarity_top_k * 2
                )

                # Fusion Retriever  (Combine both sparse and dense)
                fusion_retriever = QueryFusionRetriever(
                    [bm25_retriever, vector_retriever],
                    similarity_top_k=self.config.retrieval_config.similarity_top_k, 
                    num_queries=1, 
                    mode="reciprocal_rerank",
                    use_async=False,
                    verbose=True,
                    llm=self.llm
                )

                # 4. Reranker
                if self.config.retrieval_config.use_reranking:
                    reranker = SentenceTransformerRerank(
                        model=self.config.retrieval_config.rerank_model,
                        top_n=self.config.retrieval_config.rerank_top_n,
                        device="mps"
                    )
                    node_postprocessors = [reranker]
                else:
                    node_postprocessors = []

                # 5. Query Engine
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=fusion_retriever,
                    llm=self.llm,
                    node_postprocessors=node_postprocessors
                )

                # Create tools
                tool = QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name=f"{book.name}_Tool",
                        description=f"Query engine for {book.name}, key_words: {','.join(book.keywords)}",
                    ),
                )
                self.tools.append(tool)
                self.book_names.append(book.name)
                logger.info(f"Built tool for {book.name}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to build tool for {book.name}: {e}")
    
    def _create_agent(self):
        logger.info("Creating ReAct Agent...")

        short_term_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        self.agent = ReActAgent(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=5,
            system_prompt=(
                f"You are an intelligent research assistant. "
                "Your primary role is to synthesize answers from the provided context. "
                "You may be given a set of retrieved documents and a question. "
                "Answer the question using ONLY the provided documents. "
                "If no documents are provided or they are insufficient, say 'I cannot answer from the available resources'."
                "Do NOT attempt to use tools if context is already provided."
            ),
            memory=short_term_memory,
        )
    def _initialize_agentic_layer(self):
        logger.info("Initializing agentic layer...")
        self.orchestrator = AgentOrchestrator(
            llm = self.llm,
            agent = self.agent,
            embed_model = self.embed_model,
            book_configs = self.config.books,
            tools = self.tools
        )
    async def query(self, question: str):
        return await self.orchestrator.query(question)

    def evaluate(self, records, output_path: Optional[str] = None):
        logger.info("Evaluating records...")
        eval_results = ragas_evaluate(records)
        
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Create result entry
                entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "scores": dict(eval_results)
                }
                
                # Load existing history
                history = []
                if os.path.exists(output_path):
                    try:
                        with open(output_path, 'r') as f:
                            history = json.load(f)
                            if not isinstance(history, list):
                                history = []
                    except json.JSONDecodeError:
                        history = []
                
                history.append(entry)
                
                with open(output_path, 'w') as f:
                    json.dump(history, f, indent=2)
                    
                logger.info(f"Saved evaluation results to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")
                
        return eval_results

    async def answer(self, question_path: str, output_path: str):
        """
        Batch answer questions from a file and prepare records for evaluation.
        """
        logger.info(f"Loading questions from {question_path}...")
        try:
            with open(question_path, 'r') as f:
                questions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            return []

        if not questions:
            logger.warning("No questions found.")
            return []

        records = []
        records_to_save = []
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        for i, q in enumerate(questions):
            question_text = q['question']
            ground_truth = q['answer']
            
            logger.info(f"Processing Query {i+1}/{len(questions)}: {question_text}")
            
            try:
                # Run the agent pipeline
                result = await self.query(question_text)
                
                # Extract fields for RAGAS
                record = {
                    "question": question_text,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "ground_truth": ground_truth,
                }
                records.append(record)
                record_to_save = {
                    "question": question_text,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "ground_truth": ground_truth,
                    # Optional: Store extra metadata if needed for debugging
                    "metadata": {
                        "plan": result.get("plan"),
                        "books_used": result.get("books_used"),
                        "validation": result.get("validation"),
                        "retries": result.get("retries")
                    }
                }
                records_to_save.append(record_to_save)
            except Exception as e:
                logger.error(f"Error processing query '{question_text}': {e}")
                continue

        # Save generated answers
        try:
            # We save a format similar to what utils.answer_questions produces, 
            # or simply the records list which contains everything.
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            to_save = [{"timestamp": time_stamp}]+ records_to_save
            data = []
            if os.path.exists(output_path):
                with open(output_path,"r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    
            if isinstance(data, list):
                data.append(to_save) 
            else:
                data = [to_save]

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved generated answers to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")

        return records

async def main():
    setup_logger()
    book_config_path = "./configs/books_config.yaml"
    model_config_path = "./configs/models_config.yaml"
    
    rag_agentic = Librarian(book_config_path, model_config_path)
    evaluate_records = await rag_agentic.answer("./files/BuildTrap/eval_questions.json", "./files/BuildTrap/generated_answer/agentic_rag.json")
    evaluate_results = rag_agentic.evaluate(evaluate_records)
    print(evaluate_results)
if __name__ == "__main__":
    asyncio.run(main())