# -*- coding: utf-8 -*-
"""
Semantic Graph Integration with AI Research Agent
Connects the semantic graph to all components of the research agent
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .graph_core import SemanticGraph, GraphNode, GraphEdge, NodeType, EdgeType
from .graph_ingestion import GraphIngestionEngine, IngestionSource
from .graph_retrieval import GraphAwareRetrieval, RetrievalStrategy
from .graph_planning import GraphAwarePlanning, PlanningStrategy
from .graph_rlhf import GraphRLHFIntegration, PreferenceType
from .graph_monitoring import GraphMonitoring

logger = logging.getLogger(__name__)

class SemanticGraphAgent:
    """Main integration class for semantic graph with research agent"""
    
    def __init__(self, use_neo4j: bool = False, neo4j_config: Optional[Dict[str, Any]] = None):
        # Initialize core graph
        self.graph = SemanticGraph(use_neo4j, neo4j_config)
        
        # Initialize components
        self.ingestion_engine = GraphIngestionEngine(self.graph)
        self.retrieval_system = GraphAwareRetrieval(self.graph)
        self.planning_system = GraphAwarePlanning(self.graph)
        self.rlhf_integration = GraphRLHFIntegration(self.graph)
        self.monitoring = GraphMonitoring(self.graph)
        
        # Integration state
        self.is_initialized = False
        self.sync_hooks_enabled = True
        
        logger.info("Semantic graph agent initialized")
    
    def initialize_with_training_data(self, training_contexts: List[str]):
        """Initialize graph with training data"""
        logger.info(f"Initializing semantic graph with {len(training_contexts)} training contexts")
        
        # Ingest training contexts
        for context in training_contexts:
            self.ingestion_engine.ingest_event(
                IngestionSource.MEMORY,
                {
                    'content': context,
                    'source': 'training',
                    'importance': 0.7
                },
                priority=2
            )
        
        # Process ingestion queue
        self.ingestion_engine.process_queue()
        
        self.is_initialized = True
        logger.info("Semantic graph initialization completed")
    
    # Memory integration hooks
    def on_memory_write(self, memory_data: Dict[str, Any]):
        """Hook called after memory writes"""
        if not self.sync_hooks_enabled:
            return
            
        self.ingestion_engine.ingest_event(
            IngestionSource.MEMORY,
            memory_data,
            priority=2
        )
    
    # Planning integration hooks  
    def on_plan_execution(self, plan_data: Dict[str, Any]):
        """Hook called after plan executions"""
        if not self.sync_hooks_enabled:
            return
            
        self.ingestion_engine.ingest_event(
            IngestionSource.PLANNER_OUTPUTS,
            plan_data,
            priority=2
        )    

    # Enhanced retrieval with graph awareness
    def enhanced_retrieval(self, query: str, query_embedding: Optional[Any] = None,
                          strategy: str = "hybrid", top_k: int = 10,
                          node_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced retrieval using graph-aware methods"""
        
        # Convert string strategy to enum
        strategy_enum = RetrievalStrategy.HYBRID
        if strategy in ['vector_only', 'graph_only', 'hybrid', 'graph_expansion', 'neighborhood_boost']:
            strategy_enum = RetrievalStrategy(strategy)
        
        # Convert node type strings to enums
        node_type_enums = None
        if node_types:
            node_type_enums = []
            for nt in node_types:
                try:
                    node_type_enums.append(NodeType(nt))
                except ValueError:
                    continue
        
        # Perform retrieval
        results = self.retrieval_system.retrieve(
            query, query_embedding, strategy_enum, top_k, node_type_enums
        )
        
        # Format results for return
        formatted_results = []
        for result in results:
            formatted_results.append({
                'node_id': result.node.id,
                'label': result.node.label,
                'type': result.node.type.value,
                'score': result.score,
                'method': result.retrieval_method,
                'explanation': result.explanation,
                'properties': result.node.properties,
                'graph_path': result.graph_path,
                'neighborhood_boost': result.neighborhood_boost
            })
        
        return {
            'query': query,
            'strategy': strategy,
            'results': formatted_results,
            'total_results': len(formatted_results)
        }
    
    # Enhanced planning with graph guidance
    def enhanced_planning(self, research_question: str, context: Dict[str, Any] = None,
                         strategy: str = "hybrid", max_steps: int = 8) -> Dict[str, Any]:
        """Enhanced planning using graph-aware methods"""
        
        # Convert string strategy to enum
        strategy_enum = PlanningStrategy.HYBRID
        if strategy in ['standard', 'graph_guided', 'neighborhood_seeded', 'relevance_weighted', 'hybrid']:
            strategy_enum = PlanningStrategy(strategy)
        
        # Generate plan
        plan = self.planning_system.generate_plan(
            research_question, context, strategy_enum, max_steps
        )
        
        # Optimize plan order
        if 'plan_steps' in plan:
            plan['plan_steps'] = self.planning_system.optimize_plan_order(plan['plan_steps'])
        
        return plan
    
    # RLHF integration methods
    def record_user_feedback(self, user_id: str, preferred_content: str, 
                           rejected_content: str = "", feedback_type: str = "quality",
                           confidence: float = 1.0, context: str = "") -> str:
        """Record user feedback in the graph"""
        return self.rlhf_integration.record_feedback(
            user_id, preferred_content, rejected_content, 
            feedback_type, confidence, context
        )
    
    def get_generation_guidance(self, user_id: str, content_type: str = "general") -> Dict[str, Any]:
        """Get generation guidance based on user preferences"""
        return self.rlhf_integration.get_generation_guidance(user_id, content_type)
    
    def detect_reward_hacking(self, user_id: str = None) -> Dict[str, Any]:
        """Detect reward hacking patterns"""
        return self.rlhf_integration.detect_reward_hacking(user_id)
    
    # Tool usage tracking
    def track_tool_usage(self, tool_name: str, method: str = "", dataset: str = "", 
                        result: str = "", purpose: str = ""):
        """Track tool usage in the graph"""
        self.ingestion_engine.ingest_event(
            IngestionSource.TOOL_USAGE,
            {
                'tool_name': tool_name,
                'method': method,
                'dataset': dataset,
                'result': result,
                'purpose': purpose,
                'timestamp': datetime.now().isoformat()
            },
            priority=1
        )
    
    # Research findings integration
    def record_research_finding(self, finding: str, confidence: float = 0.5,
                              sources: List[str] = None, step_info: Dict[str, Any] = None):
        """Record a research finding in the graph"""
        self.ingestion_engine.ingest_event(
            IngestionSource.RESEARCH_FINDINGS,
            {
                'finding': finding,
                'confidence': confidence,
                'sources': sources or [],
                'step_info': step_info or {},
                'timestamp': datetime.now().isoformat()
            },
            priority=2
        )
    
    # Context engineering integration
    def record_context_event(self, context_type: str, content: str, relevance: float = 0.5):
        """Record context engineering event"""
        self.ingestion_engine.ingest_event(
            IngestionSource.CONTEXT_ENGINEERING,
            {
                'context_type': context_type,
                'content': content,
                'relevance': relevance,
                'timestamp': datetime.now().isoformat()
            },
            priority=1
        )
    
    # Diffusion integration
    def record_diffusion_output(self, diffusion_type: str, content: str, 
                              creativity_score: float = 0.5, method: str = "unknown"):
        """Record diffusion model output"""
        self.ingestion_engine.ingest_event(
            IngestionSource.DIFFUSION_OUTPUTS,
            {
                'type': diffusion_type,
                'content': content,
                'creativity_score': creativity_score,
                'method': method,
                'timestamp': datetime.now().isoformat()
            },
            priority=1
        )
    
    # Monitoring and statistics
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        graph_stats = self.monitoring.collect_comprehensive_stats()
        
        return {
            'graph_core': {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'node_types': graph_stats.node_stats['by_type'],
                'edge_types': graph_stats.edge_stats['by_type']
            },
            'ingestion': self.ingestion_engine.get_ingestion_statistics(),
            'retrieval': self.retrieval_system.get_retrieval_statistics(),
            'planning': self.planning_system.get_planning_statistics(),
            'rlhf': self.rlhf_integration.get_rlhf_statistics(),
            'monitoring': {
                'health_score': graph_stats.health_metrics['health_score'],
                'health_status': graph_stats.health_metrics['status'],
                'connectivity': graph_stats.connectivity_stats,
                'performance': graph_stats.performance_stats
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return self.monitoring.get_monitoring_dashboard_data()
    
    def process_pending_ingestion(self):
        """Process any pending ingestion events"""
        self.ingestion_engine.process_queue()
    
    def cleanup(self):
        """Cleanup resources"""
        self.graph.cleanup()
        logger.info("Semantic graph agent cleanup completed")

# Factory function for easy integration
def create_semantic_graph_agent(use_neo4j: bool = False, 
                               neo4j_config: Optional[Dict[str, Any]] = None) -> SemanticGraphAgent:
    """Factory function to create a semantic graph agent"""
    return SemanticGraphAgent(use_neo4j, neo4j_config)