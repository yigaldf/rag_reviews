"""RAG Builder - Handles index building and management"""
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import faiss
import tiktoken
import json
from openai import OpenAI

logger = logging.getLogger('rag_system.builder')


class RAGBuilder:
    """Builds and manages FAISS index for RAG"""
    
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        
    def build_or_load(self) -> Tuple[faiss.Index, List[Dict]]:
        """Main entry: build new or load existing index"""
        index_path = self.config.index_dir / f"faiss_{self.config.num_samples}.index"
        meta_path = self.config.index_dir / f"meta_{self.config.num_samples}.jsonl"
        
        if index_path.exists() and meta_path.exists():
            logger.info("Loading existing index...")
            return self.load_artifacts()
        else:
            logger.info("Building new index from scratch...")
            return self.build_from_scratch()
    
    def build_from_scratch(self) -> Tuple[faiss.Index, List[Dict]]:
        """Complete build pipeline"""
        # 1. Load data
        df = self.load_data(self.config.num_samples)
        
        # 2. Chunk texts
        texts = df['Review_Text'].astype(str).tolist()
        chunks = self.chunk_texts(texts)
        
        # 3. Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # 4. Build FAISS index
        index, meta = self.build_index(embeddings, chunks, df)
        
        # 5. Save artifacts
        self.save_artifacts(index, meta)
        
        return index, meta
    
    def load_data(self, num_samples: int) -> pd.DataFrame:
        """Load review data from CSV"""
        logger.info(f"Loading {num_samples} samples...")
        
        csv_path = self.config.data_dir / "DisneylandReviews.csv"
        reviews_all = pd.read_csv(csv_path, encoding='latin1')
        
        samples = reviews_all.head(num_samples)
        df = samples[['Review_ID', 'Branch', 'Reviewer_Location', 
                      'Year_Month', 'Rating', 'Review_Text']].dropna(subset=['Review_Text']).copy()
        
        logger.info(f"Loaded {len(df)} reviews")
        return df
    
    def chunk_texts(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Split texts into overlapping chunks"""
        logger.info(f"Chunking {len(texts)} documents...")
        
        enc = tiktoken.encoding_for_model(self.config.embed_model)
        chunks = []
        total_tokens = 0
        
        for doc_id, text in enumerate(texts):
            if doc_id % 1000 == 0 and doc_id > 0:
                logger.debug(f"Chunking progress: {doc_id}/{len(texts)} documents")
            
            toks = enc.encode(text)
            total_tokens += len(toks)
            
            step = self.config.max_tokens - self.config.overlap
            for start in range(0, len(toks), step):
                piece = enc.decode(toks[start:start + self.config.max_tokens])
                chunks.append((doc_id, piece))
        
        logger.info(f"Created {len(chunks)} chunks from {len(texts)} documents")
        logger.info(f"Total tokens: {total_tokens:,}, Avg per doc: {total_tokens/len(texts):.1f}")
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Tuple[int, str]]) -> np.ndarray:
        """Generate embeddings for chunks (with caching)"""
        # Check cache first
        cache_path = self.config.index_dir / f"embeddings_{self.config.num_samples}.npy"
        
        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        chunk_texts = [c[1] for c in chunks]
        vecs = []
        batch_size = self.config.batch_size
        total_batches = (len(chunk_texts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for batch_idx, i in enumerate(range(0, len(chunk_texts), batch_size), 1):
            batch = chunk_texts[i:i + batch_size]
            
            try:
                resp = self.client.embeddings.create(model=self.config.embed_model, input=batch)
                vecs.extend([r.embedding for r in resp.data])
                
                if batch_idx % 10 == 0 or batch_idx == total_batches:
                    elapsed = time.time() - start_time
                    rate = (batch_idx * batch_size) / elapsed if elapsed > 0 else 0
                    logger.info(f"Embedding progress: {batch_idx}/{total_batches} batches "
                              f"({len(vecs)}/{len(chunk_texts)} texts, {rate:.1f} texts/sec)")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                raise
        
        embeddings = np.array(vecs, dtype='float32')
        
        # Cache for future use
        self.config.index_dir.mkdir(exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Tuple[int, str]], 
                    df: pd.DataFrame) -> Tuple[faiss.Index, List[Dict]]:
        """Build FAISS index and metadata"""
        logger.info(f"Building FAISS index for {len(embeddings)} embeddings...")
        
        # Build index
        dim = embeddings.shape[1]
        logger.debug(f"Embedding dimension: {dim}")
        
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        # Build metadata
        logger.info("Building metadata...")
        meta = []
        for doc_id, chunk_text in chunks:
            r = df.iloc[doc_id]
            meta.append({
                "row_id": int(doc_id),
                "review_id": int(r['Review_ID']) if pd.notna(r['Review_ID']) else None,
                "branch": str(r['Branch']),
                "reviewer_location": str(r['Reviewer_Location']),
                "year_month": str(r['Year_Month']),
                "rating": float(r['Rating']) if pd.notna(r['Rating']) else None,
                "chunk": chunk_text
            })
        
        return index, meta
    
    def save_artifacts(self, index: faiss.Index, meta: List[Dict]):
        """Save index and metadata to disk"""
        self.config.index_dir.mkdir(exist_ok=True)
        
        index_path = self.config.index_dir / f"faiss_{self.config.num_samples}.index"
        meta_path = self.config.index_dir / f"meta_{self.config.num_samples}.jsonl"
        
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, str(index_path))
        
        logger.info(f"Saving metadata to {meta_path}")
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        
        logger.info("Artifacts saved successfully")
    
    def load_artifacts(self) -> Tuple[faiss.Index, List[Dict]]:
        """Load existing index and metadata"""
        index_path = self.config.index_dir / f"faiss_{self.config.num_samples}.index"
        meta_path = self.config.index_dir / f"meta_{self.config.num_samples}.jsonl"
        
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors")
        
        logger.info(f"Loading metadata from {meta_path}")
        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        
        logger.info(f"Metadata loaded: {len(meta)} entries")
        
        return index, meta

