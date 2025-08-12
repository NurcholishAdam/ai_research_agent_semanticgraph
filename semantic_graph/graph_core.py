# -*- coding: utf-8 -*-
"""
Core Semantic Graph Implementation
Provides the foundational graph structure with nodes, edges, and operations
"""

import networkx as nx
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the semantic graph"""
    CONCEPT = "concept"
    PAPER = "paper"
    FINDING = "finding"
    METHOD = "method"
    TOOL = "tool"
    MODEL = "model"
    TASK = "task"
    SUBTASK = "subtask"
    PREFERENCE = "preference"
    STYLE = "style"
    DATASET = "dataset"
    METRIC = "metric"
    ENTITY = "entity"
    ARTIFACT = "artifact"

class EdgeType(Enum):
    """Types of edges in the semantic graph"""
    # Citation and reference relationships
    CITES = "cites"
    MENTIONS = "mentions"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    
    # Tool and method relationships
    USES = "uses"
    IMPLEMENTS = "implements"
    ENHANCES = "enhances"
    
    # Task and planning relationships
    DECOMPOSES_INTO = "decomposes_into"
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    
    # Preference and RLHF relationships
    PREFERRED_STYLE = "preferred_style"
    DOWNVOTED_FOR = "downvoted_for"
    IMPROVES = "improves"
    
    # Data and evaluation relationships
    EVALUATED_ON = "evaluated_on"
    MEASURED_BY = "measured_by"
    
    # General semantic relationships
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"

@dataclass
class GraphNode:
    """Represents a node in the semantic graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType = NodeType.CONCEPT
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            'id': self.id,
            'type': self.type.value,
            'label': self.label,
            'properties': self.properties,
            'embedding': self.embedding,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'access_count': self.access_count,
            'importance_score': self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        """Create node from dictionary representation"""
        node = cls(
            id=data['id'],
            type=NodeType(data['type']),
            label=data['label'],
            properties=data.get('properties', {}),
            embedding=data.get('embedding'),
            access_count=data.get('access_count', 0),
            importance_score=data.get('importance_score', 0.5)
        )
        
        if 'created_at' in data:
            node.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            node.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return node

@dataclass
class GraphEdge:
    """Represents an edge in the semantic graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: EdgeType = EdgeType.RELATED_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type.value,
            'properties': self.properties,
            'weight': self.weight,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        """Create edge from dictionary representation"""
        edge = cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            type=EdgeType(data['type']),
            properties=data.get('properties', {}),
            weight=data.get('weight', 1.0),
            confidence=data.get('confidence', 1.0)
        )
        
        if 'created_at' in data:
            edge.created_at = datetime.fromisoformat(data['created_at'])
            
        return edge

class SemanticGraph:
    """Main semantic graph implementation using NetworkX"""
    
    def __init__(self, use_neo4j: bool = False, neo4j_config: Optional[Dict[str, Any]] = None):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        
        # Neo4j integration (optional)
        self.use_neo4j = use_neo4j
        self.neo4j_driver = None
        
        if use_neo4j and neo4j_config:
            self._initialize_neo4j(neo4j_config)
        
        # Graph statistics
        self.stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'last_updated': datetime.now()
        }
        
        logger.info(f"Semantic graph initialized with {'Neo4j' if use_neo4j else 'NetworkX'} backend")
    
    def _initialize_neo4j(self, config: Dict[str, Any]):
        """Initialize Neo4j connection (optional)"""
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(
                config['uri'],
                auth=(config['username'], config['password'])
            )
            logger.info("Neo4j connection established")
        except ImportError:
            logger.warning("Neo4j driver not available, falling back to NetworkX only")
            self.use_neo4j = False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.use_neo4j = False
    
    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.to_dict())
        
        if self.use_neo4j:
            self._add_node_to_neo4j(node)
        
        self.stats['nodes_created'] += 1
        self.stats['last_updated'] = datetime.now()
        
        logger.debug(f"Added node {node.id} of type {node.type.value}")
        return node.id
    
    def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the graph"""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Source or target node not found for edge {edge.id}")
        
        self.edges[edge.id] = edge
        self.graph.add_edge(
            edge.source_id, 
            edge.target_id, 
            key=edge.id,
            **edge.to_dict()
        )
        
        if self.use_neo4j:
            self._add_edge_to_neo4j(edge)
        
        self.stats['edges_created'] += 1
        self.stats['last_updated'] = datetime.now()
        
        logger.debug(f"Added edge {edge.id} of type {edge.type.value}")
        return edge.id
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        node = self.nodes.get(node_id)
        if node:
            node.access_count += 1
            node.updated_at = datetime.now()
        return node
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """Get an edge by ID"""
        return self.edges.get(edge_id)
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None, 
                     max_depth: int = 1) -> List[GraphNode]:
        """Get neighboring nodes with optional filtering"""
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth or current_id in visited:
                continue
                
            visited.add(current_id)
            
            # Get direct neighbors
            for neighbor_id in self.graph.neighbors(current_id):
                if neighbor_id not in visited:
                    # Check edge type filtering
                    if edge_types:
                        edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                        if edge_data:
                            edge_matches = any(
                                EdgeType(edge_info.get('type', '')) in edge_types
                                for edge_info in edge_data.values()
                            )
                            if not edge_matches:
                                continue
                    
                    neighbor_node = self.nodes.get(neighbor_id)
                    if neighbor_node and depth > 0:  # Don't include the source node
                        neighbors.append(neighbor_node)
                    
                    if depth + 1 < max_depth:
                        queue.append((neighbor_id, depth + 1))
        
        return neighbors
    
    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source_id, target_id, cutoff=max_length
            ))
            return paths[:10]  # Limit to top 10 paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_subgraph(self, node_ids: List[str], include_edges: bool = True) -> 'SemanticGraph':
        """Extract a subgraph containing specified nodes"""
        subgraph = SemanticGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges if requested
        if include_edges:
            for edge in self.edges.values():
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    subgraph.add_edge(edge)
        
        return subgraph
    
    def search_nodes(self, query: str, node_types: Optional[List[NodeType]] = None, 
                    limit: int = 10) -> List[GraphNode]:
        """Search nodes by label and properties"""
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            # Type filtering
            if node_types and node.type not in node_types:
                continue
            
            # Text matching
            score = 0
            if query_lower in node.label.lower():
                score += 2
            
            # Search in properties
            for key, value in node.properties.items():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 1
            
            if score > 0:
                results.append((node, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results[:limit]]
    
    def get_node_importance(self, node_id: str) -> float:
        """Calculate node importance based on graph structure"""
        if node_id not in self.nodes:
            return 0.0
        
        # Combine multiple importance metrics
        centrality = nx.degree_centrality(self.graph).get(node_id, 0)
        pagerank = nx.pagerank(self.graph).get(node_id, 0)
        access_score = min(1.0, self.nodes[node_id].access_count / 100)
        
        importance = (centrality * 0.4 + pagerank * 0.4 + access_score * 0.2)
        
        # Update stored importance
        self.nodes[node_id].importance_score = importance
        return importance
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        node_type_counts = {}
        edge_type_counts = {}
        
        for node in self.nodes.values():
            node_type_counts[node.type.value] = node_type_counts.get(node.type.value, 0) + 1
        
        for edge in self.edges.values():
            edge_type_counts[edge.type.value] = edge_type_counts.get(edge.type.value, 0) + 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_type_counts,
            'edge_types': edge_type_counts,
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            'creation_stats': self.stats
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export entire graph to dictionary"""
        return {
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'stats': self.stats
        }
    
    def import_from_dict(self, data: Dict[str, Any]):
        """Import graph from dictionary"""
        # Clear existing graph
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Import nodes
        for node_data in data.get('nodes', []):
            node = GraphNode.from_dict(node_data)
            self.add_node(node)
        
        # Import edges
        for edge_data in data.get('edges', []):
            edge = GraphEdge.from_dict(edge_data)
            self.add_edge(edge)
        
        # Import stats
        if 'stats' in data:
            self.stats.update(data['stats'])
    
    def _add_node_to_neo4j(self, node: GraphNode):
        """Add node to Neo4j database"""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            session.run(
                f"CREATE (n:{node.type.value} {{id: $id, label: $label, properties: $properties}})",
                id=node.id,
                label=node.label,
                properties=json.dumps(node.properties)
            )
    
    def _add_edge_to_neo4j(self, edge: GraphEdge):
        """Add edge to Neo4j database"""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            session.run(
                f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                f"CREATE (a)-[r:{edge.type.value} {{id: $id, weight: $weight, confidence: $confidence}}]->(b)",
                source_id=edge.source_id,
                target_id=edge.target_id,
                id=edge.id,
                weight=edge.weight,
                confidence=edge.confidence
            )
    
    def cleanup(self):
        """Cleanup resources"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")