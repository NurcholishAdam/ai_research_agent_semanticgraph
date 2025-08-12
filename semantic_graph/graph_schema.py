# -*- coding: utf-8 -*-
"""
Graph Schema and Entity Extraction
Defines the schema and provides tools for extracting entities and generating triples
"""

import re
import spacy
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from datetime import datetime

from .graph_core import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

# Try to load spaCy model, fallback to basic extraction if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (OSError, ImportError):
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model not available, using basic entity extraction")

@dataclass
class Triple:
    """Represents a semantic triple (subject, predicate, object)"""
    subject: str
    predicate: EdgeType
    object: str
    confidence: float = 1.0
    source: str = ""
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class GraphSchema:
    """Defines the semantic graph schema and validation rules"""
    
    def __init__(self):
        # Define valid node type transitions and relationships
        self.valid_edges = {
            NodeType.CONCEPT: [EdgeType.RELATED_TO, EdgeType.PART_OF, EdgeType.MENTIONS],
            NodeType.PAPER: [EdgeType.CITES, EdgeType.MENTIONS, EdgeType.SUPPORTS, EdgeType.CONTRADICTS],
            NodeType.FINDING: [EdgeType.SUPPORTS, EdgeType.CONTRADICTS, EdgeType.MENTIONS],
            NodeType.METHOD: [EdgeType.USES, EdgeType.IMPLEMENTS, EdgeType.ENHANCES],
            NodeType.TOOL: [EdgeType.USES, EdgeType.IMPLEMENTS, EdgeType.ENHANCES],
            NodeType.MODEL: [EdgeType.USES, EdgeType.IMPLEMENTS, EdgeType.EVALUATED_ON],
            NodeType.TASK: [EdgeType.DECOMPOSES_INTO, EdgeType.DEPENDS_ON, EdgeType.ENABLES],
            NodeType.SUBTASK: [EdgeType.DEPENDS_ON, EdgeType.PART_OF],
            NodeType.PREFERENCE: [EdgeType.PREFERRED_STYLE, EdgeType.IMPROVES],
            NodeType.STYLE: [EdgeType.PREFERRED_STYLE, EdgeType.RELATED_TO],
            NodeType.DATASET: [EdgeType.EVALUATED_ON, EdgeType.USES],
            NodeType.METRIC: [EdgeType.MEASURED_BY, EdgeType.IMPROVES]
        }
        
        # Define node type hierarchies
        self.node_hierarchies = {
            NodeType.CONCEPT: [NodeType.ENTITY, NodeType.FINDING],
            NodeType.METHOD: [NodeType.TOOL, NodeType.MODEL],
            NodeType.TASK: [NodeType.SUBTASK],
            NodeType.PREFERENCE: [NodeType.STYLE]
        }
        
        # Define required properties for each node type
        self.required_properties = {
            NodeType.PAPER: ['title', 'authors'],
            NodeType.FINDING: ['content', 'confidence'],
            NodeType.METHOD: ['description', 'category'],
            NodeType.TOOL: ['name', 'purpose'],
            NodeType.TASK: ['description', 'priority'],
            NodeType.PREFERENCE: ['user_id', 'preference_type'],
            NodeType.DATASET: ['name', 'size'],
            NodeType.METRIC: ['name', 'unit']
        }
    
    def validate_node(self, node: GraphNode) -> Tuple[bool, List[str]]:
        """Validate a node against the schema"""
        errors = []
        
        # Check required properties
        required_props = self.required_properties.get(node.type, [])
        for prop in required_props:
            if prop not in node.properties:
                errors.append(f"Missing required property '{prop}' for node type {node.type.value}")
        
        # Validate property types
        if node.type == NodeType.FINDING and 'confidence' in node.properties:
            confidence = node.properties['confidence']
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append("Confidence must be a number between 0 and 1")
        
        return len(errors) == 0, errors
    
    def validate_edge(self, edge: GraphEdge, source_node: GraphNode, target_node: GraphNode) -> Tuple[bool, List[str]]:
        """Validate an edge against the schema"""
        errors = []
        
        # Check if edge type is valid for source node type
        valid_edge_types = self.valid_edges.get(source_node.type, [])
        if edge.type not in valid_edge_types:
            errors.append(f"Edge type {edge.type.value} not valid for source node type {source_node.type.value}")
        
        # Validate specific edge constraints
        if edge.type == EdgeType.CITES and source_node.type != NodeType.PAPER:
            errors.append("CITES edge can only originate from PAPER nodes")
        
        if edge.type == EdgeType.DECOMPOSES_INTO and target_node.type != NodeType.SUBTASK:
            errors.append("DECOMPOSES_INTO edge must target SUBTASK nodes")
        
        return len(errors) == 0, errors

class EntityExtractor:
    """Extracts entities from text to create graph nodes"""
    
    def __init__(self):
        self.citation_patterns = [
            r'\b[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)',  # Author et al. (year)
            r'\b[A-Z][a-z]+\s+\(\d{4}\)',  # Author (year)
            r'doi:\s*10\.\d+\/[^\s]+',  # DOI
            r'arXiv:\d+\.\d+',  # arXiv
            r'https?://[^\s]+'  # URLs
        ]
        
        self.method_keywords = [
            'algorithm', 'method', 'approach', 'technique', 'framework',
            'model', 'system', 'tool', 'library', 'implementation'
        ]
        
        self.concept_keywords = [
            'concept', 'theory', 'principle', 'hypothesis', 'assumption',
            'definition', 'property', 'characteristic', 'feature'
        ]
    
    def extract_entities(self, text: str, source: str = "") -> List[Tuple[str, NodeType, Dict[str, Any]]]:
        """Extract entities from text and classify them"""
        entities = []
        
        # Extract citations
        citations = self._extract_citations(text)
        for citation in citations:
            entities.append((citation, NodeType.PAPER, {
                'citation': citation,
                'source': source,
                'extracted_at': datetime.now().isoformat()
            }))
        
        # Extract methods and tools
        methods = self._extract_methods(text)
        for method in methods:
            entities.append((method, NodeType.METHOD, {
                'description': method,
                'category': 'extracted',
                'source': source
            }))
        
        # Extract concepts using NLP if available
        if SPACY_AVAILABLE:
            concepts = self._extract_concepts_nlp(text)
        else:
            concepts = self._extract_concepts_basic(text)
        
        for concept in concepts:
            entities.append((concept, NodeType.CONCEPT, {
                'definition': concept,
                'source': source,
                'confidence': 0.8
            }))
        
        # Extract datasets and metrics
        datasets = self._extract_datasets(text)
        for dataset in datasets:
            entities.append((dataset, NodeType.DATASET, {
                'name': dataset,
                'source': source
            }))
        
        return entities
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return list(set(citations))  # Remove duplicates
    
    def _extract_methods(self, text: str) -> List[str]:
        """Extract methods and tools from text"""
        methods = []
        text_lower = text.lower()
        
        # Look for method keywords in context
        for keyword in self.method_keywords:
            pattern = rf'\b\w*{keyword}\w*\b'
            matches = re.findall(pattern, text_lower)
            methods.extend(matches)
        
        # Look for capitalized method names
        method_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        potential_methods = re.findall(method_pattern, text)
        
        # Filter for likely method names
        for method in potential_methods:
            if any(keyword in method.lower() for keyword in self.method_keywords):
                methods.append(method)
        
        return list(set(methods))
    
    def _extract_concepts_nlp(self, text: str) -> List[str]:
        """Extract concepts using spaCy NLP"""
        doc = nlp(text)
        concepts = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                concepts.append(ent.text)
        
        # Extract noun phrases that might be concepts
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to short phrases
                concepts.append(chunk.text)
        
        return list(set(concepts))
    
    def _extract_concepts_basic(self, text: str) -> List[str]:
        """Basic concept extraction without NLP"""
        concepts = []
        
        # Look for concept keywords
        for keyword in self.concept_keywords:
            pattern = rf'\b\w*{keyword}\w*\b'
            matches = re.findall(pattern, text.lower())
            concepts.extend(matches)
        
        # Extract capitalized terms
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        capitalized_terms = re.findall(capitalized_pattern, text)
        concepts.extend(capitalized_terms)
        
        return list(set(concepts))
    
    def _extract_datasets(self, text: str) -> List[str]:
        """Extract dataset names from text"""
        dataset_patterns = [
            r'\b[A-Z]+[-_]?\d*\b',  # Common dataset naming patterns
            r'\b\w+\s+dataset\b',
            r'\b\w+\s+corpus\b',
            r'\b\w+\s+benchmark\b'
        ]
        
        datasets = []
        for pattern in dataset_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            datasets.extend(matches)
        
        return list(set(datasets))

class TripleGenerator:
    """Generates semantic triples from various sources"""
    
    def __init__(self):
        self.relationship_patterns = {
            EdgeType.CITES: [
                r'(\w+(?:\s+et\s+al\.)?)\s+\(\d{4}\)\s+(?:shows?|demonstrates?|proves?|argues?)',
                r'(?:according\s+to|as\s+shown\s+by)\s+(\w+(?:\s+et\s+al\.)?)\s+\(\d{4}\)'
            ],
            EdgeType.USES: [
                r'(?:using|with|via|through)\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)',
                r'([A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:is\s+used|was\s+applied)'
            ],
            EdgeType.IMPROVES: [
                r'(\w+)\s+(?:improves?|enhances?|outperforms?)\s+(\w+)',
                r'(\w+)\s+(?:is\s+better\s+than|superior\s+to)\s+(\w+)'
            ]
        }
    
    def generate_triples_from_text(self, text: str, source: str = "") -> List[Triple]:
        """Generate triples from natural language text"""
        triples = []
        
        # Extract relationship patterns
        for edge_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) == 2:
                        subject, obj = match.groups()
                        triples.append(Triple(
                            subject=subject.strip(),
                            predicate=edge_type,
                            object=obj.strip(),
                            confidence=0.7,
                            source=source
                        ))
        
        return triples
    
    def generate_triples_from_memory(self, memory_content: str, citations: List[str], 
                                   concepts: List[str]) -> List[Triple]:
        """Generate triples from memory content"""
        triples = []
        
        # Create triples between memory content and citations
        for citation in citations:
            triples.append(Triple(
                subject=f"memory_{hash(memory_content) % 10000}",
                predicate=EdgeType.CITES,
                object=citation,
                confidence=0.9,
                source="memory"
            ))
        
        # Create triples between memory content and concepts
        for concept in concepts:
            triples.append(Triple(
                subject=f"memory_{hash(memory_content) % 10000}",
                predicate=EdgeType.MENTIONS,
                object=concept,
                confidence=0.8,
                source="memory"
            ))
        
        # Create concept relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                triples.append(Triple(
                    subject=concept1,
                    predicate=EdgeType.RELATED_TO,
                    object=concept2,
                    confidence=0.6,
                    source="memory"
                ))
        
        return triples
    
    def generate_triples_from_planning(self, task: str, subtasks: List[str], 
                                     dependencies: List[Tuple[str, str]]) -> List[Triple]:
        """Generate triples from planning information"""
        triples = []
        
        # Task decomposition triples
        for subtask in subtasks:
            triples.append(Triple(
                subject=task,
                predicate=EdgeType.DECOMPOSES_INTO,
                object=subtask,
                confidence=1.0,
                source="planning"
            ))
        
        # Dependency triples
        for dep_source, dep_target in dependencies:
            triples.append(Triple(
                subject=dep_source,
                predicate=EdgeType.DEPENDS_ON,
                object=dep_target,
                confidence=1.0,
                source="planning"
            ))
        
        return triples
    
    def generate_triples_from_rlhf(self, user_id: str, preferred_content: str, 
                                 rejected_content: str, feedback_type: str) -> List[Triple]:
        """Generate triples from RLHF feedback"""
        triples = []
        
        # Create preference nodes
        preference_id = f"pref_{user_id}_{hash(preferred_content) % 10000}"
        
        # Preference triples
        triples.append(Triple(
            subject=user_id,
            predicate=EdgeType.PREFERRED_STYLE,
            object=preference_id,
            confidence=1.0,
            source="rlhf",
            properties={'feedback_type': feedback_type}
        ))
        
        # Rejection triples
        if rejected_content:
            rejection_id = f"reject_{user_id}_{hash(rejected_content) % 10000}"
            triples.append(Triple(
                subject=user_id,
                predicate=EdgeType.DOWNVOTED_FOR,
                object=rejection_id,
                confidence=1.0,
                source="rlhf",
                properties={'feedback_type': feedback_type}
            ))
        
        return triples
    
    def generate_triples_from_tools(self, tool_name: str, method_used: str, 
                                  dataset_processed: str = None) -> List[Triple]:
        """Generate triples from tool usage"""
        triples = []
        
        # Tool-method relationship
        if method_used:
            triples.append(Triple(
                subject=tool_name,
                predicate=EdgeType.IMPLEMENTS,
                object=method_used,
                confidence=0.9,
                source="tools"
            ))
        
        # Tool-dataset relationship
        if dataset_processed:
            triples.append(Triple(
                subject=tool_name,
                predicate=EdgeType.USES,
                object=dataset_processed,
                confidence=0.8,
                source="tools"
            ))
        
        return triples