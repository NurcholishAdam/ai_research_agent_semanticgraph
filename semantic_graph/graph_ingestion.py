# -*- coding: utf-8 -*-
"""
Graph Ingestion Engine
Handles ingestion of data from various sources into the semantic graph
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .graph_core import SemanticGraph, GraphNode, GraphEdge, NodeType, EdgeType
from .graph_schema import EntityExtractor, TripleGenerator, Triple

logger = logging.getLogger(__name__)

class IngestionSource(Enum):
    """Sources of data for graph ingestion"""
    MEMORY = "memory"
    RETRIEVAL_LOGS = "retrieval_logs"
    PLANNER_OUTPUTS = "planner_outputs"
    RLHF_FEEDBACK = "rlhf_feedback"
    TOOL_USAGE = "tool_usage"
    RESEARCH_FINDINGS = "research_findings"
    CONTEXT_ENGINEERING = "context_engineering"
    DIFFUSION_OUTPUTS = "diffusion_outputs"

@dataclass
class IngestionEvent:
    """Represents an ingestion event"""
    source: IngestionSource
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high
    processed: bool = False

class GraphIngestionEngine:
    """Main engine for ingesting data into the semantic graph"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.entity_extractor = EntityExtractor()
        self.triple_generator = TripleGenerator()
        
        # Ingestion queue and processing
        self.ingestion_queue: List[IngestionEvent] = []
        self.processing_hooks: Dict[IngestionSource, List[Callable]] = {}
        self.batch_size = 10
        self.max_workers = 4
        
        # Statistics
        self.ingestion_stats = {
            'events_processed': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'errors': 0,
            'last_processed': None
        }
        
        # Register default processing hooks
        self._register_default_hooks()
        
        logger.info("Graph ingestion engine initialized")
    
    def _register_default_hooks(self):
        """Register default processing hooks for each source type"""
        self.register_hook(IngestionSource.MEMORY, self._process_memory_event)
        self.register_hook(IngestionSource.RETRIEVAL_LOGS, self._process_retrieval_event)
        self.register_hook(IngestionSource.PLANNER_OUTPUTS, self._process_planning_event)
        self.register_hook(IngestionSource.RLHF_FEEDBACK, self._process_rlhf_event)
        self.register_hook(IngestionSource.TOOL_USAGE, self._process_tool_event)
        self.register_hook(IngestionSource.RESEARCH_FINDINGS, self._process_research_event)
        self.register_hook(IngestionSource.CONTEXT_ENGINEERING, self._process_context_event)
        self.register_hook(IngestionSource.DIFFUSION_OUTPUTS, self._process_diffusion_event)
    
    def register_hook(self, source: IngestionSource, hook: Callable):
        """Register a processing hook for a specific source"""
        if source not in self.processing_hooks:
            self.processing_hooks[source] = []
        self.processing_hooks[source].append(hook)
        logger.debug(f"Registered hook for {source.value}")
    
    def ingest_event(self, source: IngestionSource, data: Dict[str, Any], priority: int = 1):
        """Add an ingestion event to the queue"""
        event = IngestionEvent(
            source=source,
            data=data,
            timestamp=datetime.now(),
            priority=priority
        )
        
        self.ingestion_queue.append(event)
        logger.debug(f"Added ingestion event from {source.value}")
        
        # Process immediately if high priority
        if priority >= 3:
            self.process_queue()
    
    def process_queue(self, batch_size: Optional[int] = None):
        """Process events in the ingestion queue"""
        if not self.ingestion_queue:
            return
        
        batch_size = batch_size or self.batch_size
        
        # Sort by priority and timestamp
        self.ingestion_queue.sort(key=lambda x: (-x.priority, x.timestamp))
        
        # Process batch
        batch = self.ingestion_queue[:batch_size]
        self.ingestion_queue = self.ingestion_queue[batch_size:]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_event, event) for event in batch]
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing ingestion event: {e}")
                    self.ingestion_stats['errors'] += 1
        
        self.ingestion_stats['last_processed'] = datetime.now()
        logger.info(f"Processed {len(batch)} ingestion events")
    
    def _process_event(self, event: IngestionEvent):
        """Process a single ingestion event"""
        try:
            hooks = self.processing_hooks.get(event.source, [])
            
            for hook in hooks:
                hook(event)
            
            event.processed = True
            self.ingestion_stats['events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing event from {event.source.value}: {e}")
            raise
    
    def _process_memory_event(self, event: IngestionEvent):
        """Process memory-related events"""
        data = event.data
        content = data.get('content', '')
        citations = data.get('citations', [])
        concepts = data.get('concepts', [])
        memory_id = data.get('memory_id', '')
        
        # Create memory node
        memory_node = GraphNode(
            id=memory_id or f"memory_{hash(content) % 10000}",
            type=NodeType.FINDING,
            label=content[:100] + "..." if len(content) > 100 else content,
            properties={
                'content': content,
                'confidence': data.get('importance', 0.5),
                'source': 'memory',
                'timestamp': event.timestamp.isoformat()
            }
        )
        
        node_id = self.graph.add_node(memory_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Generate and add triples
        triples = self.triple_generator.generate_triples_from_memory(content, citations, concepts)
        self._add_triples_to_graph(triples, event.source.value)
    
    def _process_retrieval_event(self, event: IngestionEvent):
        """Process retrieval log events"""
        data = event.data
        query = data.get('query', '')
        results = data.get('results', [])
        retrieval_method = data.get('method', 'unknown')
        
        # Create query node
        query_node = GraphNode(
            id=f"query_{hash(query) % 10000}",
            type=NodeType.CONCEPT,
            label=query,
            properties={
                'query_text': query,
                'retrieval_method': retrieval_method,
                'timestamp': event.timestamp.isoformat(),
                'result_count': len(results)
            }
        )
        
        self.graph.add_node(query_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Create edges to retrieved content
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            result_id = result.get('id', f"result_{i}")
            
            # Create result node if it doesn't exist
            if not self.graph.get_node(result_id):
                result_node = GraphNode(
                    id=result_id,
                    type=NodeType.FINDING,
                    label=result.get('content', '')[:100],
                    properties={
                        'content': result.get('content', ''),
                        'similarity_score': result.get('similarity', 0.0),
                        'source': 'retrieval'
                    }
                )
                self.graph.add_node(result_node)
                self.ingestion_stats['nodes_created'] += 1
            
            # Create retrieval edge
            edge = GraphEdge(
                source_id=query_node.id,
                target_id=result_id,
                type=EdgeType.RELATED_TO,
                weight=result.get('similarity', 0.5),
                confidence=0.8,
                properties={'retrieval_rank': i + 1}
            )
            
            self.graph.add_edge(edge)
            self.ingestion_stats['edges_created'] += 1
    
    def _process_planning_event(self, event: IngestionEvent):
        """Process planning output events"""
        data = event.data
        task = data.get('task', '')
        subtasks = data.get('subtasks', [])
        dependencies = data.get('dependencies', [])
        plan_id = data.get('plan_id', f"plan_{hash(task) % 10000}")
        
        # Create task node
        task_node = GraphNode(
            id=plan_id,
            type=NodeType.TASK,
            label=task,
            properties={
                'description': task,
                'priority': data.get('priority', 1),
                'estimated_effort': data.get('estimated_effort', 1.0),
                'planning_method': data.get('method', 'standard'),
                'timestamp': event.timestamp.isoformat()
            }
        )
        
        self.graph.add_node(task_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Create subtask nodes and relationships
        subtask_ids = []
        for i, subtask in enumerate(subtasks):
            subtask_id = f"{plan_id}_subtask_{i}"
            subtask_node = GraphNode(
                id=subtask_id,
                type=NodeType.SUBTASK,
                label=subtask.get('description', '') if isinstance(subtask, dict) else subtask,
                properties={
                    'description': subtask.get('description', '') if isinstance(subtask, dict) else subtask,
                    'order': i,
                    'parent_task': plan_id
                }
            )
            
            self.graph.add_node(subtask_node)
            subtask_ids.append(subtask_id)
            self.ingestion_stats['nodes_created'] += 1
            
            # Create decomposition edge
            decomp_edge = GraphEdge(
                source_id=plan_id,
                target_id=subtask_id,
                type=EdgeType.DECOMPOSES_INTO,
                weight=1.0,
                confidence=1.0
            )
            
            self.graph.add_edge(decomp_edge)
            self.ingestion_stats['edges_created'] += 1
        
        # Generate and add planning triples
        triples = self.triple_generator.generate_triples_from_planning(task, subtasks, dependencies)
        self._add_triples_to_graph(triples, event.source.value)
    
    def _process_rlhf_event(self, event: IngestionEvent):
        """Process RLHF feedback events"""
        data = event.data
        user_id = data.get('user_id', 'anonymous')
        feedback_type = data.get('feedback_type', 'rating')
        preferred_content = data.get('preferred_content', '')
        rejected_content = data.get('rejected_content', '')
        rating = data.get('rating', 0)
        
        # Create user preference node
        pref_id = f"pref_{user_id}_{hash(preferred_content) % 10000}"
        pref_node = GraphNode(
            id=pref_id,
            type=NodeType.PREFERENCE,
            label=f"User {user_id} preference",
            properties={
                'user_id': user_id,
                'preference_type': feedback_type,
                'rating': rating,
                'preferred_content': preferred_content,
                'rejected_content': rejected_content,
                'timestamp': event.timestamp.isoformat()
            }
        )
        
        self.graph.add_node(pref_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Generate and add RLHF triples
        triples = self.triple_generator.generate_triples_from_rlhf(
            user_id, preferred_content, rejected_content, feedback_type
        )
        self._add_triples_to_graph(triples, event.source.value)
    
    def _process_tool_event(self, event: IngestionEvent):
        """Process tool usage events"""
        data = event.data
        tool_name = data.get('tool_name', '')
        method_used = data.get('method', '')
        dataset_processed = data.get('dataset', '')
        result = data.get('result', '')
        
        # Create tool node
        tool_node = GraphNode(
            id=f"tool_{tool_name}",
            type=NodeType.TOOL,
            label=tool_name,
            properties={
                'name': tool_name,
                'purpose': data.get('purpose', ''),
                'usage_count': data.get('usage_count', 1),
                'last_used': event.timestamp.isoformat()
            }
        )
        
        # Add or update tool node
        existing_tool = self.graph.get_node(tool_node.id)
        if existing_tool:
            existing_tool.properties['usage_count'] = existing_tool.properties.get('usage_count', 0) + 1
            existing_tool.properties['last_used'] = event.timestamp.isoformat()
        else:
            self.graph.add_node(tool_node)
            self.ingestion_stats['nodes_created'] += 1
        
        # Generate and add tool triples
        triples = self.triple_generator.generate_triples_from_tools(tool_name, method_used, dataset_processed)
        self._add_triples_to_graph(triples, event.source.value)
    
    def _process_research_event(self, event: IngestionEvent):
        """Process research findings events"""
        data = event.data
        finding = data.get('finding', '')
        confidence = data.get('confidence', 0.5)
        sources = data.get('sources', [])
        
        # Extract entities from the finding
        entities = self.entity_extractor.extract_entities(finding, "research")
        
        # Create nodes for extracted entities
        for entity_text, node_type, properties in entities:
            entity_node = GraphNode(
                id=f"{node_type.value}_{hash(entity_text) % 10000}",
                type=node_type,
                label=entity_text,
                properties=properties
            )
            
            # Check if node already exists
            if not self.graph.get_node(entity_node.id):
                self.graph.add_node(entity_node)
                self.ingestion_stats['nodes_created'] += 1
        
        # Generate triples from the finding text
        triples = self.triple_generator.generate_triples_from_text(finding, "research")
        self._add_triples_to_graph(triples, event.source.value)
    
    def _process_context_event(self, event: IngestionEvent):
        """Process context engineering events"""
        data = event.data
        context_type = data.get('context_type', 'general')
        context_content = data.get('content', '')
        relevance_score = data.get('relevance', 0.5)
        
        # Create context node
        context_node = GraphNode(
            id=f"context_{hash(context_content) % 10000}",
            type=NodeType.CONCEPT,
            label=f"{context_type} context",
            properties={
                'context_type': context_type,
                'content': context_content,
                'relevance_score': relevance_score,
                'source': 'context_engineering',
                'timestamp': event.timestamp.isoformat()
            }
        )
        
        self.graph.add_node(context_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Extract entities and create relationships
        entities = self.entity_extractor.extract_entities(context_content, "context")
        for entity_text, node_type, properties in entities:
            entity_id = f"{node_type.value}_{hash(entity_text) % 10000}"
            
            # Create context-entity relationship
            edge = GraphEdge(
                source_id=context_node.id,
                target_id=entity_id,
                type=EdgeType.MENTIONS,
                weight=relevance_score,
                confidence=0.7
            )
            
            if self.graph.get_node(entity_id):  # Only add edge if target exists
                self.graph.add_edge(edge)
                self.ingestion_stats['edges_created'] += 1
    
    def _process_diffusion_event(self, event: IngestionEvent):
        """Process diffusion model outputs"""
        data = event.data
        diffusion_type = data.get('type', 'generation')
        generated_content = data.get('content', '')
        creativity_score = data.get('creativity_score', 0.5)
        
        # Create diffusion output node
        diffusion_node = GraphNode(
            id=f"diffusion_{hash(generated_content) % 10000}",
            type=NodeType.ARTIFACT,
            label=f"Diffusion {diffusion_type}",
            properties={
                'diffusion_type': diffusion_type,
                'content': generated_content,
                'creativity_score': creativity_score,
                'generation_method': data.get('method', 'unknown'),
                'timestamp': event.timestamp.isoformat()
            }
        )
        
        self.graph.add_node(diffusion_node)
        self.ingestion_stats['nodes_created'] += 1
        
        # Extract entities and create relationships
        entities = self.entity_extractor.extract_entities(generated_content, "diffusion")
        for entity_text, node_type, properties in entities:
            entity_id = f"{node_type.value}_{hash(entity_text) % 10000}"
            
            # Create diffusion-entity relationship
            edge = GraphEdge(
                source_id=diffusion_node.id,
                target_id=entity_id,
                type=EdgeType.MENTIONS,
                weight=creativity_score,
                confidence=0.6
            )
            
            if self.graph.get_node(entity_id):
                self.graph.add_edge(edge)
                self.ingestion_stats['edges_created'] += 1
    
    def _add_triples_to_graph(self, triples: List[Triple], source: str):
        """Add triples to the graph as nodes and edges"""
        for triple in triples:
            # Create or get subject node
            subject_id = f"entity_{hash(triple.subject) % 10000}"
            if not self.graph.get_node(subject_id):
                subject_node = GraphNode(
                    id=subject_id,
                    type=NodeType.CONCEPT,
                    label=triple.subject,
                    properties={'source': source}
                )
                self.graph.add_node(subject_node)
                self.ingestion_stats['nodes_created'] += 1
            
            # Create or get object node
            object_id = f"entity_{hash(triple.object) % 10000}"
            if not self.graph.get_node(object_id):
                object_node = GraphNode(
                    id=object_id,
                    type=NodeType.CONCEPT,
                    label=triple.object,
                    properties={'source': source}
                )
                self.graph.add_node(object_node)
                self.ingestion_stats['nodes_created'] += 1
            
            # Create edge
            edge = GraphEdge(
                source_id=subject_id,
                target_id=object_id,
                type=triple.predicate,
                weight=triple.confidence,
                confidence=triple.confidence,
                properties=triple.properties
            )
            
            self.graph.add_edge(edge)
            self.ingestion_stats['edges_created'] += 1
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return {
            'queue_size': len(self.ingestion_queue),
            'processing_stats': self.ingestion_stats,
            'hooks_registered': {source.value: len(hooks) for source, hooks in self.processing_hooks.items()},
            'batch_size': self.batch_size,
            'max_workers': self.max_workers
        }
    
    def clear_queue(self):
        """Clear the ingestion queue"""
        cleared_count = len(self.ingestion_queue)
        self.ingestion_queue.clear()
        logger.info(f"Cleared {cleared_count} events from ingestion queue")