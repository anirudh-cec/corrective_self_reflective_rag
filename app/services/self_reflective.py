from app.services.llm_service import LLMService
from app.models import ReflectionResult, SelfReflectiveResult, RetrievedChunk
from app.config import get_settings
from datetime import datetime
from loguru import logger
import json


class SelfReflectiveService:
    def __init__(self):
        self.settings = get_settings()
        self.llm = LLMService()
    
    def generate_initial_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        """Generate initial answer from retrieved chunks"""
        
        context = "\n\n".join([
            f"Document {i+1}:\n{chunk.content}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""Answer the following query using the provided documents.

Query: {query}

Documents:
{context}

Provide a clear, accurate answer."""
        
        system_prompt = "You are a helpful assistant that answers questions based on provided context."
        
        return self.llm.generate(prompt, system_prompt, max_tokens=500)
    
    def reflect_on_answer(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> ReflectionResult:
        """Reflect on generated answer to check grounding"""
        
        context = "\n\n".join([
            f"Document {i+1}: {chunk.content}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""Evaluate if the generated answer is properly grounded in the provided documents.

Query: {query}

Generated Answer:
{answer}

Source Documents:
{context}

Analyze and provide evaluation as JSON:
{{
    "answer_grounded": <true|false>,
    "hallucination_detected": <true|false>,
    "reflection_score": <float 0.0-1.0>,
    "sources_cited": [<list of document numbers that support the answer>],
    "reflection_reason": "<explanation>",
    "needs_regeneration": <true|false>
}}

Criteria:
- answer_grounded: Are all claims in the answer supported by the documents?
- hallucination_detected: Does the answer include information NOT in the documents?
- reflection_score: Overall quality (1.0 = perfect, 0.0 = completely unsupported)
- needs_regeneration: Should we retry with refined retrieval?
"""
        
        system_prompt = "You are an answer evaluator for RAG systems. Always respond with valid JSON."
        
        try:
            response = self.llm.generate_with_json(prompt, system_prompt)
            reflection_data = json.loads(response)
            
            return ReflectionResult(
                answer_grounded=reflection_data.get("answer_grounded", False),
                hallucination_detected=reflection_data.get("hallucination_detected", True),
                sources_cited=reflection_data.get("sources_cited", []),
                reflection_score=reflection_data.get("reflection_score", 0.5),
                needs_regeneration=reflection_data.get("needs_regeneration", True),
                reflection_reason=reflection_data.get("reflection_reason", "Unknown"),
                reflected_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return ReflectionResult(
                answer_grounded=False,
                hallucination_detected=True,
                sources_cited=[],
                reflection_score=0.5,
                needs_regeneration=True,
                reflection_reason=f"Reflection failed: {str(e)}",
                reflected_at=datetime.utcnow()
            )
    
    def execute_self_reflective(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        retrieval_function
    ) -> SelfReflectiveResult:
        """Execute Self-Reflective RAG pipeline"""
        
        iterations = 0
        max_iterations = self.settings.max_reflection_retries
        
        current_chunks = retrieved_chunks
        final_answer = None
        final_reflection = None
        
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Self-Reflective iteration {iterations}")
            
            # Generate answer
            answer = self.generate_initial_answer(query, current_chunks)
            
            # Reflect on answer
            reflection = self.reflect_on_answer(query, answer, current_chunks)
            
            logger.info(f"Reflection score: {reflection.reflection_score:.2f}, Grounded: {reflection.answer_grounded}")
            
            # Check if answer is good enough
            if reflection.reflection_score >= self.settings.reflection_min_score:
                final_answer = answer
                final_reflection = reflection
                logger.info("Self-Reflective: Answer approved")
                break
            
            # If not good and we have retries left, refine and retry
            if reflection.needs_regeneration and iterations < max_iterations:
                logger.info("Self-Reflective: Refining query and retrying")
                refined_query = self._refine_query(query, reflection)
                current_chunks = retrieval_function(refined_query)
            else:
                final_answer = answer
                final_reflection = reflection
                break
        
        if final_answer is None:
            final_answer = answer
            final_reflection = reflection
        
        return SelfReflectiveResult(
            final_answer=final_answer,
            iterations=iterations,
            reflection=final_reflection,
            retrieved_chunks=current_chunks
        )
    
    def _refine_query(self, original_query: str, reflection: ReflectionResult) -> str:
        """Refine query based on reflection feedback"""
        
        prompt = f"""The original query didn't retrieve good enough information. Refine the query to get better results.

Original Query: {original_query}

Reflection Feedback: {reflection.reflection_reason}

Provide a refined query that might retrieve more relevant information. Return only the refined query text, nothing else."""
        
        system_prompt = "You are a query refinement expert for retrieval systems."
        
        try:
            refined = self.llm.generate(prompt, system_prompt, max_tokens=100)
            logger.info(f"Refined query: {refined}")
            return refined.strip()
        except Exception as e:
            logger.error(f"Query refinement error: {e}")
            return original_query
