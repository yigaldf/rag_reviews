"""RAG Query Handler - Handles query processing and answer generation"""
import logging
import time
from typing import List, Dict, Any
import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger('rag_system.query')


class RAGQueryHandler:
    """Handles query processing and answer generation"""
    
    def __init__(self, config, index: faiss.Index, metadata: List[Dict]):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.index = index
        self.metadata = metadata
        logger.info("RAGQueryHandler initialized")
    
    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k similar chunks for a query"""
        logger.debug(f"Retrieving top-{k} results for query: '{query[:100]}...'")
        
        # Embed query
        logger.debug("Embedding query...")
        qv = np.array(
            [self.client.embeddings.create(
                model=self.config.embed_model, 
                input=query
            ).data[0].embedding],
            dtype='float32'
        )
        
        # Search FAISS index
        logger.debug("Searching FAISS index...")
        D, I = self.index.search(qv, k)
        
        logger.info(f"Top result distance: {D[0][0]:.4f}")
        
        # Format results
        results = []
        for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
            m = self.metadata[int(idx)]
            results.append({
                "rank": rank,
                "distance": float(dist),
                "branch": m["branch"],
                "rating": m["rating"],
                "reviewer_location": m["reviewer_location"],
                "snippet": m["chunk"][:300].replace("\n", " ")
            })
            
            logger.debug(f"Rank {rank}: distance={dist:.4f}, branch={m['branch']}, rating={m['rating']}")
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""You are a helpful assistant answering strictly from the provided Disneyland reviews.
Question: {query}

Use ONLY the context below; if unsure, say you don't know.
Context:
{context}
"""
        return prompt
    
    def answer_query(
        self,
        query: str,
        k: int = None,
        temperature: float = None,
        model: str = None
    ) -> Dict[str, Any]:
        """Main query processing method"""
        # Use defaults if not specified
        k = k or self.config.default_k
        temperature = temperature or self.config.default_temperature
        model = model or self.config.llm_model
        
        logger.info(f"Processing query: '{query[:100]}...'")
        logger.debug(f"Parameters: k={k}, temperature={temperature}, model={model}")
        
        try:
            # 1. Retrieve relevant chunks
            retrieval_results = self.retrieve(query, k)
            
            if not retrieval_results:
                logger.warning("No results retrieved")
                return {
                    "query": query,
                    "answer": "No relevant information found.",
                    "retrieval_results": [],
                    "k": k,
                    "temperature": temperature,
                    "model": model
                }
            
            # 2. Build context
            context = "\n\n".join([
                f"[{i+1}] Branch: {r['branch']}, Rating: {r['rating']}/5\n{r['snippet']}"
                for i, r in enumerate(retrieval_results)
            ])
            
            # 3. Build prompt
            prompt = self.build_prompt(query, context)
            
            logger.debug(f"Context length: {len(context)} chars")
            
            # 4. Generate answer
            logger.debug(f"Generating answer with {model}...")
            start_time = time.time()
            
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            answer = resp.choices[0].message.content
            generation_time = time.time() - start_time
            
            logger.info(f"Answer generated in {generation_time:.2f}s")
            logger.debug(f"Answer length: {len(answer)} chars")
            
            return {
                "query": query,
                "answer": answer,
                "retrieval_results": retrieval_results,
                "k": k,
                "temperature": temperature,
                "model": model
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in index"""
        return self.index.ntotal if self.index else 0

