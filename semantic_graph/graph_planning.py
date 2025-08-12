# -*- coding: utf-8 -*-
"""
Graph-Aware Planning System
Uses semantic graph to enhance planning with connected nodes and relevance priors
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import random
from collections import defaultdict

from .graph_core import SemanticGraph, GraphNode, NodeType, EdgeType

logger = logging.getLogger(__name__)

class PlanningStrategy(Enum):
    """Different planning strategies"""
    STANDARD = "standard"
    GRAPH_GUIDED = "graph_guided"
    NEIGHBORHOOD_SEEDED = "neighborhood_seeded"
    RELEVANCE_WEIGHTED = "relevance_weighted"
    HYBRID = "hybrid"

@dataclass
class PlanningNode:
    """Represents a node in the planning graph"""
    task_id: str
    description: str
    node_type: str = "task"
    priority: float = 1.0
    estimated_effort: float = 1.0
    dependencies: List[str] = None
    graph_connections: List[str] = None
    relevance_score: float = 0.5
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.graph_connections is None:
            self.graph_connections = []

class GraphAwarePlanning:
    """Main graph-aware planning system"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.planning_cache = {}
        
        # Planning parameters
        self.max_subtasks = 10
        self.relevance_threshold = 0.3
        self.neighborhood_expansion_depth = 2
        self.diversity_factor = 0.3
        
        logger.info("Graph-aware planning system initialized")
    
    def generate_plan(self, research_question: str, context: Dict[str, Any] = None,
                     strategy: PlanningStrategy = PlanningStrategy.HYBRID,
                     max_steps: int = 8) -> Dict[str, Any]:
        """Generate a research plan using graph-aware methods"""
        
        # Extract key concepts from research question
        key_concepts = self._extract_planning_concepts(research_question)
        
        # Find relevant graph nodes
        relevant_nodes = self._find_relevant_nodes(key_concepts, research_question)
        
        # Generate plan based on strategy
        if strategy == PlanningStrategy.STANDARD:
            plan_nodes = self._generate_standard_plan(research_question, max_steps)
        elif strategy == PlanningStrategy.GRAPH_GUIDED:
            plan_nodes = self._generate_graph_guided_plan(research_question, relevant_nodes, max_steps)
        elif strategy == PlanningStrategy.NEIGHBORHOOD_SEEDED:
            plan_nodes = self._generate_neighborhood_seeded_plan(research_question, relevant_nodes, max_steps)
        elif strategy == PlanningStrategy.RELEVANCE_WEIGHTED:
            plan_nodes = self._generate_relevance_weighted_plan(research_question, relevant_nodes, max_steps)
        else:  # HYBRID
            plan_nodes = self._generate_hybrid_plan(research_question, relevant_nodes, max_steps)
        
        # Build final plan structure
        plan = {
            'research_question': research_question,
            'strategy': strategy.value,
            'key_concepts': key_concepts,
            'relevant_graph_nodes': [node.id for node in relevant_nodes],
            'plan_steps': self._convert_to_plan_steps(plan_nodes),
            'estimated_complexity': self._calculate_plan_complexity(plan_nodes),
            'graph_connectivity': self._analyze_plan_connectivity(plan_nodes),
            'generation_method': 'graph_aware_planning'
        }
        
        logger.info(f"Generated {len(plan_nodes)} planning steps using {strategy.value} strategy")
        return plan
    
    def _extract_planning_concepts(self, research_question: str) -> List[str]:
        """Extract key concepts from research question for planning"""
        # Simple keyword extraction - can be enhanced with NLP
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        
        words = research_question.lower().split()
        concepts = []
        
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 3 and clean_word not in stop_words:
                concepts.append(clean_word)
        
        # Look for multi-word concepts
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 6:
                bigrams.append(bigram)
        
        return concepts + bigrams
    
    def _find_relevant_nodes(self, concepts: List[str], research_question: str) -> List[GraphNode]:
        """Find graph nodes relevant to the planning task"""
        relevant_nodes = []
        
        # Search for nodes matching concepts
        for concept in concepts:
            matching_nodes = self.graph.search_nodes(
                concept, 
                node_types=[NodeType.CONCEPT, NodeType.METHOD, NodeType.TOOL, NodeType.TASK],
                limit=5
            )
            relevant_nodes.extend(matching_nodes)
        
        # Search for nodes matching the full question
        question_nodes = self.graph.search_nodes(research_question, limit=10)
        relevant_nodes.extend(question_nodes)
        
        # Remove duplicates and sort by importance
        unique_nodes = {}
        for node in relevant_nodes:
            if node.id not in unique_nodes:
                unique_nodes[node.id] = node
        
        sorted_nodes = sorted(
            unique_nodes.values(), 
            key=lambda x: self.graph.get_node_importance(x.id), 
            reverse=True
        )
        
        return sorted_nodes[:15]  # Limit to top 15 relevant nodes
    
    def _generate_standard_plan(self, research_question: str, max_steps: int) -> List[PlanningNode]:
        """Generate a standard plan without graph guidance"""
        standard_steps = [
            "Define research scope and objectives",
            "Conduct literature review",
            "Identify key concepts and terminology",
            "Search for recent research and publications",
            "Analyze existing methodologies",
            "Synthesize findings and insights",
            "Evaluate evidence quality",
            "Generate comprehensive summary"
        ]
        
        plan_nodes = []
        for i, step in enumerate(standard_steps[:max_steps]):
            plan_nodes.append(PlanningNode(
                task_id=f"task_{i+1}",
                description=step,
                priority=1.0 - (i * 0.1),
                estimated_effort=1.0,
                dependencies=[f"task_{i}"] if i > 0 else []
            ))
        
        return plan_nodes
    
    def _generate_graph_guided_plan(self, research_question: str, 
                                   relevant_nodes: List[GraphNode], max_steps: int) -> List[PlanningNode]:
        """Generate plan guided by graph structure"""
        plan_nodes = []
        
        # Start with high-importance nodes
        seed_nodes = relevant_nodes[:3]
        
        for i, node in enumerate(seed_nodes):
            # Create task based on node type and properties
            task_description = self._node_to_task_description(node, research_question)
            
            plan_nodes.append(PlanningNode(
                task_id=f"graph_task_{i+1}",
                description=task_description,
                priority=node.importance_score,
                estimated_effort=self._estimate_task_effort(node),
                graph_connections=[node.id],
                relevance_score=node.importance_score
            ))
        
        # Add connecting tasks based on graph relationships
        for i in range(len(plan_nodes), max_steps):
            connecting_task = self._generate_connecting_task(plan_nodes, relevant_nodes, i+1)
            if connecting_task:
                plan_nodes.append(connecting_task)
        
        return plan_nodes
    
    def _generate_neighborhood_seeded_plan(self, research_question: str,
                                         relevant_nodes: List[GraphNode], max_steps: int) -> List[PlanningNode]:
        """Generate plan seeded from graph neighborhoods"""
        plan_nodes = []
        used_neighborhoods = set()
        
        for i in range(max_steps):
            if i < len(relevant_nodes):
                seed_node = relevant_nodes[i]
                
                # Get neighborhood
                neighbors = self.graph.get_neighbors(
                    seed_node.id, 
                    max_depth=self.neighborhood_expansion_depth
                )
                
                # Create task from neighborhood
                neighborhood_key = frozenset([n.id for n in neighbors[:3]])
                if neighborhood_key not in used_neighborhoods:
                    task_description = self._neighborhood_to_task_description(
                        seed_node, neighbors, research_question
                    )
                    
                    plan_nodes.append(PlanningNode(
                        task_id=f"neighborhood_task_{i+1}",
                        description=task_description,
                        priority=seed_node.importance_score,
                        estimated_effort=min(2.0, len(neighbors) * 0.2),
                        graph_connections=[seed_node.id] + [n.id for n in neighbors[:2]],
                        relevance_score=self._calculate_neighborhood_relevance(seed_node, neighbors)
                    ))
                    
                    used_neighborhoods.add(neighborhood_key)
        
        return plan_nodes
    
    def _generate_relevance_weighted_plan(self, research_question: str,
                                        relevant_nodes: List[GraphNode], max_steps: int) -> List[PlanningNode]:
        """Generate plan weighted by node relevance scores"""
        plan_nodes = []
        
        # Group nodes by type for diverse planning
        nodes_by_type = defaultdict(list)
        for node in relevant_nodes:
            nodes_by_type[node.type].append(node)
        
        # Generate tasks from each type
        type_priorities = {
            NodeType.CONCEPT: 0.9,
            NodeType.METHOD: 0.8,
            NodeType.TOOL: 0.7,
            NodeType.PAPER: 0.6,
            NodeType.TASK: 0.5
        }
        
        task_counter = 1
        for node_type, nodes in nodes_by_type.items():
            if task_counter > max_steps:
                break
                
            priority = type_priorities.get(node_type, 0.5)
            
            for node in nodes[:2]:  # Max 2 tasks per type
                if task_counter > max_steps:
                    break
                    
                task_description = self._create_relevance_weighted_task(node, research_question)
                
                plan_nodes.append(PlanningNode(
                    task_id=f"relevance_task_{task_counter}",
                    description=task_description,
                    priority=priority * node.importance_score,
                    estimated_effort=self._estimate_task_effort(node),
                    graph_connections=[node.id],
                    relevance_score=node.importance_score
                ))
                
                task_counter += 1
        
        return plan_nodes
    
    def _generate_hybrid_plan(self, research_question: str,
                            relevant_nodes: List[GraphNode], max_steps: int) -> List[PlanningNode]:
        """Generate hybrid plan combining multiple strategies"""
        plan_nodes = []
        
        # Start with standard foundation (30%)
        standard_count = max(1, int(max_steps * 0.3))
        standard_nodes = self._generate_standard_plan(research_question, standard_count)
        plan_nodes.extend(standard_nodes)
        
        # Add graph-guided tasks (40%)
        graph_count = max(1, int(max_steps * 0.4))
        graph_nodes = self._generate_graph_guided_plan(research_question, relevant_nodes, graph_count)
        
        # Merge and avoid duplicates
        for graph_node in graph_nodes:
            if not self._is_duplicate_task(graph_node, plan_nodes):
                plan_nodes.append(graph_node)
                if len(plan_nodes) >= max_steps:
                    break
        
        # Fill remaining with neighborhood-seeded tasks (30%)
        remaining_count = max_steps - len(plan_nodes)
        if remaining_count > 0:
            neighborhood_nodes = self._generate_neighborhood_seeded_plan(
                research_question, relevant_nodes, remaining_count
            )
            
            for neighborhood_node in neighborhood_nodes:
                if not self._is_duplicate_task(neighborhood_node, plan_nodes):
                    plan_nodes.append(neighborhood_node)
                    if len(plan_nodes) >= max_steps:
                        break
        
        return plan_nodes[:max_steps]
    
    def _node_to_task_description(self, node: GraphNode, research_question: str) -> str:
        """Convert a graph node to a task description"""
        if node.type == NodeType.CONCEPT:
            return f"Explore the concept of {node.label} in relation to {research_question}"
        elif node.type == NodeType.METHOD:
            return f"Investigate the {node.label} method and its applications"
        elif node.type == NodeType.TOOL:
            return f"Analyze how {node.label} can be used for research"
        elif node.type == NodeType.PAPER:
            return f"Review and analyze the paper: {node.label}"
        elif node.type == NodeType.TASK:
            return f"Execute task: {node.label}"
        else:
            return f"Research {node.label} and its relevance to the question"
    
    def _neighborhood_to_task_description(self, center_node: GraphNode, 
                                        neighbors: List[GraphNode], research_question: str) -> str:
        """Create task description from a node neighborhood"""
        neighbor_labels = [n.label for n in neighbors[:3]]
        
        if len(neighbor_labels) == 0:
            return f"Investigate {center_node.label} in isolation"
        elif len(neighbor_labels) == 1:
            return f"Analyze the relationship between {center_node.label} and {neighbor_labels[0]}"
        else:
            neighbors_str = ", ".join(neighbor_labels[:-1]) + f" and {neighbor_labels[-1]}"
            return f"Explore how {center_node.label} connects to {neighbors_str}"
    
    def _create_relevance_weighted_task(self, node: GraphNode, research_question: str) -> str:
        """Create task description weighted by relevance"""
        relevance_terms = ["key", "important", "relevant", "significant", "crucial"]
        relevance_term = random.choice(relevance_terms)
        
        return f"Focus on the {relevance_term} aspects of {node.label} for {research_question}"
    
    def _estimate_task_effort(self, node: GraphNode) -> float:
        """Estimate effort required for a task based on node properties"""
        base_effort = 1.0
        
        # Adjust based on node type
        type_multipliers = {
            NodeType.CONCEPT: 0.8,
            NodeType.METHOD: 1.2,
            NodeType.TOOL: 1.0,
            NodeType.PAPER: 1.5,
            NodeType.TASK: 1.0
        }
        
        effort = base_effort * type_multipliers.get(node.type, 1.0)
        
        # Adjust based on node complexity (number of connections)
        neighbors = self.graph.get_neighbors(node.id, max_depth=1)
        complexity_factor = min(2.0, 1.0 + len(neighbors) * 0.1)
        
        return effort * complexity_factor
    
    def _calculate_neighborhood_relevance(self, center_node: GraphNode, 
                                        neighbors: List[GraphNode]) -> float:
        """Calculate relevance score for a neighborhood"""
        if not neighbors:
            return center_node.importance_score
        
        # Combine center node importance with neighbor importance
        neighbor_importance = sum(n.importance_score for n in neighbors) / len(neighbors)
        
        # Weight center node more heavily
        relevance = (center_node.importance_score * 0.7 + neighbor_importance * 0.3)
        
        return min(1.0, relevance)
    
    def _is_duplicate_task(self, new_task: PlanningNode, existing_tasks: List[PlanningNode]) -> bool:
        """Check if a task is duplicate of existing tasks"""
        new_desc_words = set(new_task.description.lower().split())
        
        for existing_task in existing_tasks:
            existing_desc_words = set(existing_task.description.lower().split())
            
            # Calculate word overlap
            overlap = len(new_desc_words & existing_desc_words)
            total_words = len(new_desc_words | existing_desc_words)
            
            if total_words > 0 and overlap / total_words > 0.6:  # 60% overlap threshold
                return True
        
        return False
    
    def _convert_to_plan_steps(self, plan_nodes: List[PlanningNode]) -> List[Dict[str, Any]]:
        """Convert planning nodes to plan step format"""
        steps = []
        
        for i, node in enumerate(plan_nodes):
            step = {
                'step_id': i + 1,
                'description': node.description,
                'priority': node.priority,
                'estimated_effort': node.estimated_effort,
                'dependencies': node.dependencies,
                'graph_connections': node.graph_connections,
                'relevance_score': node.relevance_score,
                'tools_suggested': self._suggest_tools_for_task(node)
            }
            steps.append(step)
        
        return steps
    
    def _suggest_tools_for_task(self, task_node: PlanningNode) -> List[str]:
        """Suggest tools for a planning task based on graph connections"""
        suggested_tools = []
        
        # Get tools connected to the task's graph connections
        for connection_id in task_node.graph_connections:
            connected_node = self.graph.get_node(connection_id)
            if connected_node:
                # Find tool nodes connected to this node
                neighbors = self.graph.get_neighbors(
                    connection_id, 
                    edge_types=[EdgeType.USES, EdgeType.IMPLEMENTS],
                    max_depth=1
                )
                
                for neighbor in neighbors:
                    if neighbor.type == NodeType.TOOL:
                        tool_name = neighbor.label
                        if tool_name not in suggested_tools:
                            suggested_tools.append(tool_name)
        
        # Add default tools based on task description
        task_desc_lower = task_node.description.lower()
        if 'search' in task_desc_lower or 'find' in task_desc_lower:
            suggested_tools.extend(['web_search', 'academic_search'])
        if 'analyze' in task_desc_lower or 'review' in task_desc_lower:
            suggested_tools.extend(['document_processor', 'data_analysis'])
        if 'visualize' in task_desc_lower or 'chart' in task_desc_lower:
            suggested_tools.append('data_visualization')
        
        return list(set(suggested_tools))  # Remove duplicates
    
    def _calculate_plan_complexity(self, plan_nodes: List[PlanningNode]) -> float:
        """Calculate overall plan complexity"""
        if not plan_nodes:
            return 0.0
        
        # Average effort across all tasks
        avg_effort = sum(node.estimated_effort for node in plan_nodes) / len(plan_nodes)
        
        # Dependency complexity
        total_dependencies = sum(len(node.dependencies) for node in plan_nodes)
        dependency_complexity = total_dependencies / len(plan_nodes)
        
        # Graph connectivity complexity
        total_connections = sum(len(node.graph_connections) for node in plan_nodes)
        connectivity_complexity = total_connections / len(plan_nodes)
        
        # Combine factors
        complexity = (avg_effort * 0.5 + dependency_complexity * 0.3 + connectivity_complexity * 0.2)
        
        return min(3.0, complexity)  # Cap at 3.0
    
    def _analyze_plan_connectivity(self, plan_nodes: List[PlanningNode]) -> Dict[str, Any]:
        """Analyze how well the plan is connected to the graph"""
        if not plan_nodes:
            return {}
        
        total_connections = sum(len(node.graph_connections) for node in plan_nodes)
        connected_tasks = sum(1 for node in plan_nodes if node.graph_connections)
        
        # Calculate connectivity metrics
        connectivity_analysis = {
            'total_graph_connections': total_connections,
            'connected_tasks': connected_tasks,
            'connectivity_ratio': connected_tasks / len(plan_nodes),
            'avg_connections_per_task': total_connections / len(plan_nodes),
            'graph_coverage': len(set(conn for node in plan_nodes for conn in node.graph_connections))
        }
        
        return connectivity_analysis
    
    def optimize_plan_order(self, plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize the order of plan steps based on dependencies and graph structure"""
        # Create dependency graph
        step_dependencies = {}
        for step in plan_steps:
            step_id = step['step_id']
            step_dependencies[step_id] = step.get('dependencies', [])
        
        # Topological sort considering dependencies
        ordered_steps = []
        remaining_steps = plan_steps.copy()
        
        while remaining_steps:
            # Find steps with no unmet dependencies
            ready_steps = []
            for step in remaining_steps:
                step_deps = step.get('dependencies', [])
                completed_step_ids = [s['step_id'] for s in ordered_steps]
                
                if all(dep in completed_step_ids or dep == step['step_id'] for dep in step_deps):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Break circular dependencies by taking highest priority step
                ready_steps = [max(remaining_steps, key=lambda x: x.get('priority', 0))]
            
            # Sort ready steps by priority and relevance
            ready_steps.sort(key=lambda x: (x.get('priority', 0), x.get('relevance_score', 0)), reverse=True)
            
            # Add the best ready step
            best_step = ready_steps[0]
            ordered_steps.append(best_step)
            remaining_steps.remove(best_step)
        
        # Update step IDs to reflect new order
        for i, step in enumerate(ordered_steps):
            step['step_id'] = i + 1
        
        return ordered_steps
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get planning system statistics"""
        return {
            'cache_size': len(self.planning_cache),
            'parameters': {
                'max_subtasks': self.max_subtasks,
                'relevance_threshold': self.relevance_threshold,
                'neighborhood_expansion_depth': self.neighborhood_expansion_depth,
                'diversity_factor': self.diversity_factor
            },
            'graph_stats': self.graph.get_graph_statistics()
        }