# -*- coding: utf-8 -*-
"""
Graph-Aware Retrieval System
Combines vector similarity with graph neighborhood expansion for enhanced retrieval
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.metrics.pairwise import cosine_similarity

from .graph_core import SemanticGraph, GraphNode, NodeType, EdgeType

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """Different retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    GRAPH_EXPANSION = "graph_expansion"
    NEIGHBORHOOD_BOOST = "neighborhood_boost"

@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    node: GraphNode
    score: float
    retrieval_method: str
    explanation: str
    graph_path: Optional[List[str]] = None
    neighborhood_boost: float = 0.0

class GraphAwareRetrieval:
    """Main graph-aware retrieval system"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.retrieval_cache = {}
        self.cache_size_limit = 1000
        
        # Retrieval parameters
        self.vector_weight = 0.6
        self.graph_weight = 0.4
        self.expansion_depth = 2
        self.neighborhood_boost_factor = 0.3
        self.coherence_threshold = 0.7
        
        logger.info("Graph-aware retrieval system initialized")
    
    def retrieve(self, query: str, query_embedding: Optional[torch.Tensor] = None,
                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                top_k: int = 10, node_types: Optional[List[NodeType]] = None) -> List[RetrievalResult]:
        """Main retrieval method with multiple strategies"""
        
        # Check cache first
        cache_key = f"{hash(query)}_{strategy.value}_{top_k}"
        if cache_key in self.retrieval_cache:
            logger.debug("Retrieved results from cache")
            return self.retrieval_cache[cache_key]
        
        # Get candidate nodes
        candidate_nodes = self._get_candidate_nodes(node_types)
        
        if not candidate_nodes:
            return []
        
        # Apply retrieval strategy
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            results = self._vector_retrieval(query, query_embedding, candidate_nodes, top_k)
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            results = self._graph_retrieval(query, candidate_nodes, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_retrieval(query, query_embedding, candidate_nodes, top_k)
        elif strategy == RetrievalStrategy.GRAPH_EXPANSION:
            results = self._graph_expansion_retrieval(query, query_embedding, candidate_nodes, top_k)
        elif strategy == RetrievalStrategy.NEIGHBORHOOD_BOOST:
            results = self._neighborhood_boost_retrieval(query, query_embedding, candidate_nodes, top_k)
        else:
            results = self._hybrid_retrieval(query, query_embedding, candidate_nodes, top_k)
        
        # Cache results
        if len(self.retrieval_cache) >= self.cache_size_limit:
            # Remove oldest entries
            oldest_keys = list(self.retrieval_cache.keys())[:100]
            for key in oldest_keys:
                del self.retrieval_cache[key]
        
        self.retrieval_cache[cache_key] = results
        
        logger.debug(f"Retrieved {len(results)} results using {strategy.value} strategy")
        return results
    
    def _get_candidate_nodes(self, node_types: Optional[List[NodeType]] = None) -> List[GraphNode]:
        """Get candidate nodes for retrieval"""
        candidates = []
        
        for node in self.graph.nodes.values():
            if node_types is None or node.type in node_types:
                candidates.append(node)
        
        return candidates
    
    def _vector_retrieval(self, query: str, query_embedding: Optional[torch.Tensor],
                         candidates: List[GraphNode], top_k: int) -> List[RetrievalResult]:
        """Pure vector-based retrieval"""
        if query_embedding is None:
            # Fallback to text matching if no embedding
            return self._text_matching_retrieval(query, candidates, top_k)
        
        results = []
        query_vec = query_embedding.cpu().numpy().reshape(1, -1)
        
        for node in candidates:
            if node.embedding is not None:
                node_vec = np.array(node.embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vec, node_vec)[0][0]
                
                results.append(RetrievalResult(
                    node=node,
                    score=similarity,
                    retrieval_method="vector_similarity",
                    explanation=f"Vector similarity: {similarity:.3f}"
                ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _graph_retrieval(self, query: str, candidates: List[GraphNode], top_k: int) -> List[RetrievalResult]:
        """Pure graph-based retrieval using graph structure"""
        results = []
        query_lower = query.lower()
        
        # Find nodes that match the query
        matching_nodes = []
        for node in candidates:
            if query_lower in node.label.lower():
                matching_nodes.append(node)
        
        # If no direct matches, use text similarity
        if not matching_nodes:
            return self._text_matching_retrieval(query, candidates, top_k)
        
        # Expand from matching nodes using graph structure
        expanded_nodes = set()
        for node in matching_nodes:
            expanded_nodes.add(node.id)
            
            # Get neighbors
            neighbors = self.graph.get_neighbors(node.id, max_depth=self.expansion_depth)
            for neighbor in neighbors:
                expanded_nodes.add(neighbor.id)
        
        # Score nodes based on graph properties
        for node_id in expanded_nodes:
            node = self.graph.get_node(node_id)
            if node:
                # Calculate graph-based score
                importance = self.graph.get_node_importance(node_id)
                centrality_score = len(self.graph.get_neighbors(node_id)) / max(len(candidates), 1)
                
                graph_score = (importance * 0.7 + centrality_score * 0.3)
                
                results.append(RetrievalResult(
                    node=node,
                    score=graph_score,
                    retrieval_method="graph_structure",
                    explanation=f"Graph importance: {importance:.3f}, centrality: {centrality_score:.3f}"
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _hybrid_retrieval(self, query: str, query_embedding: Optional[torch.Tensor],
                         candidates: List[GraphNode], top_k: int) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector and graph methods"""
        
        # Get vector results
        vector_results = self._vector_retrieval(query, query_embedding, candidates, top_k * 2)
        vector_scores = {r.node.id: r.score for r in vector_results}
        
        # Get graph results
        graph_results = self._graph_retrieval(query, candidates, top_k * 2)
        graph_scores = {r.node.id: r.score for r in graph_results}
        
        # Combine scores
        all_node_ids = set(vector_scores.keys()) | set(graph_scores.keys())
        hybrid_results = []
        
        for node_id in all_node_ids:
            node = self.graph.get_node(node_id)
            if node:
                vector_score = vector_scores.get(node_id, 0.0)
                graph_score = graph_scores.get(node_id, 0.0)
                
                # Weighted combination
                hybrid_score = (self.vector_weight * vector_score + 
                               self.graph_weight * graph_score)
                
                hybrid_results.append(RetrievalResult(
                    node=node,
                    score=hybrid_score,
                    retrieval_method="hybrid",
                    explanation=f"Vector: {vector_score:.3f}, Graph: {graph_score:.3f}, Combined: {hybrid_score:.3f}"
                ))
        
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:top_k]
    
    def _graph_expansion_retrieval(self, query: str, query_embedding: Optional[torch.Tensor],
                                  candidates: List[GraphNode], top_k: int) -> List[RetrievalResult]:
        """Retrieval with graph neighborhood expansion"""
        
        # Start with vector retrieval to get initial candidates
        initial_results = self._vector_retrieval(query, query_embedding, candidates, top_k // 2)
        
        if not initial_results:
            return self._text_matching_retrieval(query, candidates, top_k)
        
        # Expand neighborhoods
        expanded_nodes = set()
        expansion_paths = {}
        
        for result in initial_results:
            expanded_nodes.add(result.node.id)
            expansion_paths[result.node.id] = [result.node.id]
            
            # Get neighbors with path tracking
            neighbors = self.graph.get_neighbors(result.node.id, max_depth=self.expansion_depth)
            for neighbor in neighbors:
                if neighbor.id not in expanded_nodes:
                    expanded_nodes.add(neighbor.id)
                    # Find path from initial node to neighbor
                    paths = self.graph.find_paths(result.node.id, neighbor.id, max_length=self.expansion_depth)
                    if paths:
                        expansion_paths[neighbor.id] = paths[0]
        
        # Score expanded nodes
        expansion_results = []
        for node_id in expanded_nodes:
            node = self.graph.get_node(node_id)
            if node:
                # Base score from vector similarity if available
                base_score = 0.0
                for initial_result in initial_results:
                    if initial_result.node.id == node_id:
                        base_score = initial_result.score
                        break
                
                # Graph expansion boost
                path = expansion_paths.get(node_id, [])
                path_length = len(path) - 1  # Subtract 1 for the node itself
                expansion_boost = max(0, 1.0 - (path_length * 0.2))  # Decay with distance
                
                final_score = base_score + (expansion_boost * 0.3)
                
                expansion_results.append(RetrievalResult(
                    node=node,
                    score=final_score,
                    retrieval_method="graph_expansion",
                    explanation=f"Base: {base_score:.3f}, Expansion boost: {expansion_boost:.3f}",
                    graph_path=path
                ))
        
        expansion_results.sort(key=lambda x: x.score, reverse=True)
        return expansion_results[:top_k]
    
    def _neighborhood_boost_retrieval(self, query: str, query_embedding: Optional[torch.Tensor],
                                    candidates: List[GraphNode], top_k: int) -> List[RetrievalResult]:
        """Retrieval with neighborhood coherence boosting"""
        
        # Get initial vector results
        vector_results = self._vector_retrieval(query, query_embedding, candidates, top_k * 2)
        
        if not vector_results:
            return self._text_matching_retrieval(query, candidates, top_k)
        
        # Calculate neighborhood coherence for each result
        boosted_results = []
        
        for result in vector_results:
            neighbors = self.graph.get_neighbors(result.node.id, max_depth=1)
            
            # Calculate neighborhood coherence
            coherence_score = self._calculate_neighborhood_coherence(
                result.node, neighbors, query_embedding
            )
            
            # Apply neighborhood boost
            neighborhood_boost = coherence_score * self.neighborhood_boost_factor
            boosted_score = result.score + neighborhood_boost
            
            boosted_results.append(RetrievalResult(
                node=result.node,
                score=boosted_score,
                retrieval_method="neighborhood_boost",
                explanation=f"Original: {result.score:.3f}, Coherence: {coherence_score:.3f}, Boosted: {boosted_score:.3f}",
                neighborhood_boost=neighborhood_boost
            ))
        
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results[:top_k]
    
    def _calculate_neighborhood_coherence(self, center_node: GraphNode, 
                                        neighbors: List[GraphNode],
                                        query_embedding: Optional[torch.Tensor]) -> float:
        """Calculate coherence of a node's neighborhood"""
        if not neighbors or query_embedding is None:
            return 0.0
        
        coherence_scores = []
        query_vec = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Calculate coherence with query
        for neighbor in neighbors:
            if neighbor.embedding is not None:
                neighbor_vec = np.array(neighbor.embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vec, neighbor_vec)[0][0]
                coherence_scores.append(similarity)
        
        if not coherence_scores:
            return 0.0
        
        # Return average coherence, weighted by neighborhood size
        avg_coherence = np.mean(coherence_scores)
        size_factor = min(1.0, len(neighbors) / 5.0)  # Normalize by expected neighborhood size
        
        return avg_coherence * size_factor
    
    def _text_matching_retrieval(self, query: str, candidates: List[GraphNode], 
                                top_k: int) -> List[RetrievalResult]:
        """Fallback text matching when embeddings are not available"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for node in candidates:
            # Calculate text similarity
            node_text = (node.label + " " + str(node.properties)).lower()
            node_words = set(node_text.split())
            
            # Jaccard similarity
            intersection = len(query_words & node_words)
            union = len(query_words | node_words)
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Substring matching bonus
            substring_bonus = 0.2 if query_lower in node_text else 0.0
            
            final_score = jaccard_score + substring_bonus
            
            if final_score > 0:
                results.append(RetrievalResult(
                    node=node,
                    score=final_score,
                    retrieval_method="text_matching",
                    explanation=f"Jaccard: {jaccard_score:.3f}, Substring bonus: {substring_bonus:.3f}"
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def retrieve_with_context(self, query: str, context_nodes: List[str],
                            query_embedding: Optional[torch.Tensor] = None,
                            top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve with additional context from specific nodes"""
        
        # Get standard retrieval results
        base_results = self.retrieve(query, query_embedding, RetrievalStrategy.HYBRID, top_k * 2)
        
        # Boost results that are connected to context nodes
        context_boosted_results = []
        
        for result in base_results:
            context_boost = 0.0
            
            # Check connections to context nodes
            for context_node_id in context_nodes:
                if context_node_id in self.graph.nodes:
                    # Check if there's a path between result node and context node
                    paths = self.graph.find_paths(result.node.id, context_node_id, max_length=3)
                    if paths:
                        # Boost based on shortest path length
                        shortest_path_length = min(len(path) for path in paths)
                        path_boost = max(0, 0.3 - (shortest_path_length * 0.1))
                        context_boost += path_boost
            
            boosted_score = result.score + context_boost
            
            context_boosted_results.append(RetrievalResult(
                node=result.node,
                score=boosted_score,
                retrieval_method=f"{result.retrieval_method}_context_boosted",
                explanation=f"{result.explanation}, Context boost: {context_boost:.3f}",
                neighborhood_boost=context_boost
            ))
        
        context_boosted_results.sort(key=lambda x: x.score, reverse=True)
        return context_boosted_results[:top_k]
    
    def explain_retrieval(self, query: str, result: RetrievalResult) -> Dict[str, Any]:
        """Provide detailed explanation of why a result was retrieved"""
        explanation = {
            'query': query,
            'result_node_id': result.node.id,
            'result_label': result.node.label,
            'score': result.score,
            'method': result.retrieval_method,
            'explanation': result.explanation,
            'node_properties': result.node.properties,
            'graph_analysis': {}
        }
        
        # Add graph analysis
        neighbors = self.graph.get_neighbors(result.node.id, max_depth=1)
        explanation['graph_analysis'] = {
            'neighbor_count': len(neighbors),
            'node_importance': self.graph.get_node_importance(result.node.id),
            'node_type': result.node.type.value,
            'access_count': result.node.access_count
        }
        
        if result.graph_path:
            explanation['graph_path'] = result.graph_path
        
        if result.neighborhood_boost > 0:
            explanation['neighborhood_boost'] = result.neighborhood_boost
        
        return explanation
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            'cache_size': len(self.retrieval_cache),
            'cache_limit': self.cache_size_limit,
            'parameters': {
                'vector_weight': self.vector_weight,
                'graph_weight': self.graph_weight,
                'expansion_depth': self.expansion_depth,
                'neighborhood_boost_factor': self.neighborhood_boost_factor,
                'coherence_threshold': self.coherence_threshold
            },
            'graph_stats': self.graph.get_graph_statistics()
        }
    
    def clear_cache(self):
        """Clear the retrieval cache"""
        cache_size = len(self.retrieval_cache)
        self.retrieval_cache.clear()
        logger.info(f"Cleared retrieval cache ({cache_size} entries)")