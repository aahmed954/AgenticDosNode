"""Advanced RAG system with multi-stage retrieval and reranking."""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import hashlib
import json
import time
from abc import ABC, abstractmethod

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
import tiktoken

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for different use cases."""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    HYBRID = "hybrid"  # Combine vector and keyword search
    GRAPHRAG = "graphrag"  # Graph-based RAG
    HYDE = "hyde"  # Hypothetical Document Embeddings
    FUSION = "fusion"  # RAG-Fusion with multiple queries


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_AWARE = "document_aware"
    ADAPTIVE = "adaptive"


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Retrieval
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    similarity_threshold: float = 0.7

    # Embedding
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072

    # Reranking
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Query processing
    enable_query_expansion: bool = True
    enable_query_decomposition: bool = True
    max_query_expansions: int = 3

    # Context compression
    enable_compression: bool = True
    max_context_tokens: int = 8000
    compression_ratio: float = 0.5

    # Caching
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95


@dataclass
class RetrievalResult:
    """Result from retrieval operations."""

    documents: List[Document]
    scores: List[float]
    metadata: Dict[str, Any]
    retrieval_time: float
    strategy_used: str


@dataclass
class QueryDecomposition:
    """Decomposed query for multi-hop reasoning."""

    original_query: str
    sub_queries: List[str]
    dependencies: Dict[int, List[int]]  # Query dependencies
    aggregation_method: str


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def similarity_search(
        self, query_embedding: List[float], k: int
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search."""
        pass

    @abstractmethod
    async def hybrid_search(
        self, query: str, query_embedding: List[float], k: int
    ) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining vector and keyword."""
        pass


class AdvancedRAG:
    """
    Advanced RAG system with production-ready features.

    Features:
    - Multi-stage retrieval pipeline
    - Query understanding and decomposition
    - Hybrid search with reranking
    - Context compression
    - Semantic caching
    - Advanced chunking strategies
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[RAGConfig] = None
    ):
        self.vector_store = vector_store
        self.config = config or RAGConfig()
        self.embeddings = self._initialize_embeddings()
        self.reranker = self._initialize_reranker() if config.enable_reranking else None
        self.text_splitter = self._initialize_text_splitter()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.semantic_cache: Dict[str, RetrievalResult] = {}

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        return OpenAIEmbeddings(
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimension
        )

    def _initialize_reranker(self):
        """Initialize cross-encoder for reranking."""
        if self.config.enable_reranking:
            return CrossEncoder(self.config.reranker_model)
        return None

    def _initialize_text_splitter(self) -> TextSplitter:
        """Initialize text splitter based on strategy."""

        if self.config.chunking_strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        elif self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._create_semantic_splitter()
        else:
            # Default to recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

    async def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Index documents into the vector store."""

        start_time = time.time()
        total_chunks = 0

        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Chunk documents
                chunks = []
                for doc in batch:
                    doc_chunks = await self._chunk_document(doc)
                    chunks.extend(doc_chunks)

                # Generate embeddings
                texts = [chunk.page_content for chunk in chunks]
                embeddings = await self._batch_embed(texts)

                # Add to vector store
                await self.vector_store.add_documents(chunks, embeddings)
                total_chunks += len(chunks)

                logger.info(f"Indexed batch {i//batch_size + 1}: {len(chunks)} chunks")

            indexing_time = time.time() - start_time

            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": total_chunks,
                "indexing_time": indexing_time,
                "avg_chunk_size": np.mean([len(c.page_content) for c in chunks]) if chunks else 0
            }

        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0
            }

    async def retrieve(
        self,
        query: str,
        strategy: Optional[RetrievalStrategy] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Implements multi-stage retrieval:
        1. Query understanding and expansion
        2. Initial retrieval (large set)
        3. Reranking (if enabled)
        4. Context compression
        """

        start_time = time.time()
        strategy = strategy or self.config.retrieval_strategy

        # Check semantic cache
        if self.config.enable_semantic_cache:
            cached_result = self._check_semantic_cache(query)
            if cached_result:
                logger.info("Retrieved from semantic cache")
                return cached_result

        # Query processing
        processed_queries = await self._process_query(query)

        # Execute retrieval based on strategy
        if strategy == RetrievalStrategy.HYBRID:
            results = await self._hybrid_retrieval(processed_queries, filters)
        elif strategy == RetrievalStrategy.HYDE:
            results = await self._hyde_retrieval(query, filters)
        elif strategy == RetrievalStrategy.FUSION:
            results = await self._fusion_retrieval(processed_queries, filters)
        elif strategy == RetrievalStrategy.GRAPHRAG:
            results = await self._graph_retrieval(query, filters)
        else:
            results = await self._similarity_retrieval(processed_queries[0], filters)

        # Rerank if enabled
        if self.config.enable_reranking and self.reranker:
            results = await self._rerank_results(query, results)

        # Compress context if needed
        if self.config.enable_compression:
            results = await self._compress_context(results)

        retrieval_time = time.time() - start_time

        retrieval_result = RetrievalResult(
            documents=results["documents"],
            scores=results["scores"],
            metadata={
                "strategy": strategy.value,
                "queries_processed": len(processed_queries),
                "initial_results": results.get("initial_count", len(results["documents"])),
                "final_results": len(results["documents"]),
                "reranked": self.config.enable_reranking,
                "compressed": self.config.enable_compression
            },
            retrieval_time=retrieval_time,
            strategy_used=strategy.value
        )

        # Cache result
        if self.config.enable_semantic_cache:
            self._cache_result(query, retrieval_result)

        return retrieval_result

    async def _process_query(self, query: str) -> List[str]:
        """Process and potentially expand/decompose query."""

        queries = [query]

        # Query expansion
        if self.config.enable_query_expansion:
            expanded = await self._expand_query(query)
            queries.extend(expanded[:self.config.max_query_expansions])

        # Query decomposition for complex queries
        if self.config.enable_query_decomposition and self._is_complex_query(query):
            decomposed = await self._decompose_query(query)
            queries.extend(decomposed.sub_queries)

        return queries

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""

        # Simple expansion strategy
        # In production, this would use LLM or specialized expansion models
        expansions = []

        # Add question variations
        if query.endswith("?"):
            base_query = query[:-1]
            expansions.append(f"What is {base_query}")
            expansions.append(f"Explain {base_query}")
            expansions.append(f"Define {base_query}")

        return expansions[:self.config.max_query_expansions]

    async def _decompose_query(self, query: str) -> QueryDecomposition:
        """Decompose complex query into sub-queries."""

        # Simple decomposition based on conjunctions
        # In production, use LLM for intelligent decomposition
        sub_queries = []

        if " and " in query.lower():
            parts = query.split(" and ")
            sub_queries = [part.strip() for part in parts]
        elif " or " in query.lower():
            parts = query.split(" or ")
            sub_queries = [part.strip() for part in parts]
        else:
            sub_queries = [query]

        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            dependencies={},
            aggregation_method="union"
        )

    async def _hybrid_retrieval(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform hybrid retrieval combining vector and keyword search."""

        all_documents = []
        all_scores = []

        for query in queries:
            # Get query embedding
            query_embedding = await self._embed_query(query)

            # Perform hybrid search
            results = await self.vector_store.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                k=self.config.top_k_retrieval
            )

            for doc, score in results:
                all_documents.append(doc)
                all_scores.append(score)

        # Deduplicate and aggregate scores
        unique_docs = self._deduplicate_documents(all_documents, all_scores)

        return {
            "documents": unique_docs["documents"],
            "scores": unique_docs["scores"],
            "initial_count": len(all_documents)
        }

    async def _hyde_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Hypothetical Document Embeddings (HyDE) retrieval.
        Generate hypothetical answer and search for similar documents.
        """

        # Generate hypothetical document
        # In production, this would use LLM
        hypothetical = f"The answer to '{query}' would likely discuss..."

        # Embed hypothetical document
        hyde_embedding = await self._embed_query(hypothetical)

        # Search using hypothetical embedding
        results = await self.vector_store.similarity_search(
            query_embedding=hyde_embedding,
            k=self.config.top_k_retrieval
        )

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]

        return {
            "documents": documents,
            "scores": scores,
            "hypothetical": hypothetical
        }

    async def _fusion_retrieval(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        RAG-Fusion: Retrieve using multiple query variations and fuse results.
        """

        # Retrieve for each query variation
        all_results = []
        for query in queries:
            query_embedding = await self._embed_query(query)
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=self.config.top_k_retrieval // len(queries)
            )
            all_results.append(results)

        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(all_results)

        return {
            "documents": fused_results["documents"],
            "scores": fused_results["scores"],
            "query_count": len(queries)
        }

    async def _graph_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Graph-based retrieval for structured knowledge.
        """

        # Placeholder for graph-based retrieval
        # In production, this would traverse knowledge graph
        query_embedding = await self._embed_query(query)
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=self.config.top_k_retrieval
        )

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]

        return {
            "documents": documents,
            "scores": scores,
            "graph_traversal": True
        }

    async def _similarity_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Standard similarity-based retrieval."""

        query_embedding = await self._embed_query(query)
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=self.config.top_k_retrieval
        )

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]

        # Filter by similarity threshold
        filtered_docs = []
        filtered_scores = []
        for doc, score in zip(documents, scores):
            if score >= self.config.similarity_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)

        return {
            "documents": filtered_docs,
            "scores": filtered_scores
        }

    async def _rerank_results(
        self,
        query: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rerank results using cross-encoder."""

        if not self.reranker or not results["documents"]:
            return results

        documents = results["documents"]

        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in documents]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranking scores
        sorted_indices = np.argsort(rerank_scores)[::-1][:self.config.top_k_rerank]

        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_scores = [float(rerank_scores[i]) for i in sorted_indices]

        return {
            "documents": reranked_docs,
            "scores": reranked_scores,
            "reranked": True,
            "initial_count": len(documents)
        }

    async def _compress_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context to fit within token limits."""

        documents = results["documents"]
        scores = results["scores"]

        if not documents:
            return results

        # Calculate current token count
        total_tokens = sum(
            len(self.tokenizer.encode(doc.page_content))
            for doc in documents
        )

        if total_tokens <= self.config.max_context_tokens:
            return results

        # Compress by selecting most relevant portions
        compressed_docs = []
        compressed_scores = []
        current_tokens = 0

        for doc, score in zip(documents, scores):
            doc_tokens = len(self.tokenizer.encode(doc.page_content))

            if current_tokens + doc_tokens <= self.config.max_context_tokens:
                compressed_docs.append(doc)
                compressed_scores.append(score)
                current_tokens += doc_tokens
            else:
                # Truncate document to fit
                remaining_tokens = self.config.max_context_tokens - current_tokens
                if remaining_tokens > 100:  # Minimum useful chunk
                    truncated_content = self._truncate_to_tokens(
                        doc.page_content, remaining_tokens
                    )
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata={**doc.metadata, "truncated": True}
                    )
                    compressed_docs.append(truncated_doc)
                    compressed_scores.append(score)
                break

        return {
            "documents": compressed_docs,
            "scores": compressed_scores,
            "compressed": True,
            "original_count": len(documents),
            "original_tokens": total_tokens,
            "compressed_tokens": current_tokens
        }

    async def _chunk_document(self, document: Document) -> List[Document]:
        """Chunk document based on configured strategy."""

        if self.config.chunking_strategy == ChunkingStrategy.ADAPTIVE:
            return await self._adaptive_chunking(document)
        elif self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return await self._semantic_chunking(document)
        else:
            # Use configured text splitter
            chunks = self.text_splitter.split_text(document.page_content)
            return [
                Document(
                    page_content=chunk,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                for i, chunk in enumerate(chunks)
            ]

    async def _adaptive_chunking(self, document: Document) -> List[Document]:
        """Adaptive chunking based on document structure."""

        # Analyze document structure and adapt chunk size
        content = document.page_content

        # Simple heuristic: smaller chunks for dense text
        avg_sentence_length = len(content) / max(content.count('.'), 1)

        if avg_sentence_length < 50:
            chunk_size = 500
        elif avg_sentence_length < 100:
            chunk_size = 750
        else:
            chunk_size = self.config.chunk_size

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(200, chunk_size // 5)
        )

        chunks = splitter.split_text(content)
        return [
            Document(
                page_content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_strategy": "adaptive",
                    "chunk_size": chunk_size
                }
            )
            for chunk in chunks
        ]

    async def _semantic_chunking(self, document: Document) -> List[Document]:
        """Chunk based on semantic boundaries."""

        # Placeholder for semantic chunking
        # In production, use sentence embeddings to find semantic boundaries
        return await self._adaptive_chunking(document)

    def _create_semantic_splitter(self) -> TextSplitter:
        """Create semantic-aware text splitter."""
        # Placeholder - would use sentence transformers in production
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

    async def _embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        embeddings = await self.embeddings.aembed_documents([query])
        return embeddings[0]

    async def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch."""
        return await self.embeddings.aembed_documents(texts)

    def _deduplicate_documents(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> Dict[str, Any]:
        """Deduplicate documents and aggregate scores."""

        unique_docs = {}

        for doc, score in zip(documents, scores):
            # Use content hash as key
            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()

            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = {
                    "document": doc,
                    "scores": [score],
                    "count": 1
                }
            else:
                unique_docs[doc_hash]["scores"].append(score)
                unique_docs[doc_hash]["count"] += 1

        # Aggregate scores (using max)
        final_docs = []
        final_scores = []

        for doc_data in unique_docs.values():
            final_docs.append(doc_data["document"])
            final_scores.append(max(doc_data["scores"]))

        # Sort by score
        sorted_indices = np.argsort(final_scores)[::-1]

        return {
            "documents": [final_docs[i] for i in sorted_indices],
            "scores": [final_scores[i] for i in sorted_indices]
        }

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[Document, float]]]
    ) -> Dict[str, Any]:
        """Fuse multiple result lists using Reciprocal Rank Fusion."""

        k = 60  # RRF constant
        doc_scores = {}

        for result_list in result_lists:
            for rank, (doc, _) in enumerate(result_list, 1):
                doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if doc_hash not in doc_scores:
                    doc_scores[doc_hash] = {"document": doc, "score": 0}
                doc_scores[doc_hash]["score"] += 1 / (k + rank)

        # Sort by fused score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "documents": [d["document"] for d in sorted_docs],
            "scores": [d["score"] for d in sorted_docs]
        }

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex and needs decomposition."""
        complexity_indicators = [" and ", " or ", " then ", " after ", " before "]
        return any(indicator in query.lower() for indicator in complexity_indicators)

    def _check_semantic_cache(self, query: str) -> Optional[RetrievalResult]:
        """Check if similar query exists in cache."""

        if not self.semantic_cache:
            return None

        # Simple similarity check
        # In production, use embeddings for semantic similarity
        for cached_query, result in self.semantic_cache.items():
            similarity = self._calculate_similarity(query, cached_query)
            if similarity >= self.config.cache_similarity_threshold:
                logger.info(f"Cache hit with similarity {similarity:.3f}")
                return result

        return None

    def _cache_result(self, query: str, result: RetrievalResult):
        """Cache retrieval result."""
        # Limit cache size
        if len(self.semantic_cache) > 100:
            # Remove oldest entry
            self.semantic_cache.pop(next(iter(self.semantic_cache)))

        self.semantic_cache[query] = result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple Jaccard similarity
        # In production, use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)