# -*- coding: utf-8 -*-
"""
Graph Monitoring and Statistics
Provides comprehensive monitoring and statistics for the semantic graph
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import json

from .graph_core import SemanticGraph, NodeType, EdgeType

logger = logging.getLogger(__name__)

@dataclass
class GraphStats:
    """Comprehensive graph statistics"""
    timestamp: datetime
    node_stats: Dict[str, Any]
    edge_stats: Dict[str, Any]
    connectivity_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    health_metrics: Dict[str, Any]

class GraphMonitoring:
    """Main graph monitoring system"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.stats_history = []
        self.max_history_size = 100
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'max_nodes': 10000,
            'max_edges': 50000,
            'min_connectivity': 0.1,
            'max_isolated_nodes': 100
        }
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        logger.info("Graph monitoring system initialized")
    
    def collect_comprehensive_stats(self) -> GraphStats:
        """Collect comprehensive statistics about the graph"""
        timestamp = datetime.now()
        
        # Node statistics
        node_stats = self._collect_node_stats()
        
        # Edge statistics
        edge_stats = self._collect_edge_stats()
        
        # Connectivity statistics
        connectivity_stats = self._collect_connectivity_stats()
        
        # Performance statistics
        performance_stats = self._collect_performance_stats()
        
        # Health metrics
        health_metrics = self._collect_health_metrics()
        
        stats = GraphStats(
            timestamp=timestamp,
            node_stats=node_stats,
            edge_stats=edge_stats,
            connectivity_stats=connectivity_stats,
            performance_stats=performance_stats,
            health_metrics=health_metrics
        )
        
        # Store in history
        self.stats_history.append(stats)
        if len(self.stats_history) > self.max_history_size:
            self.stats_history.pop(0)
        
        return stats
    
    def _collect_node_stats(self) -> Dict[str, Any]:
        """Collect node-related statistics"""
        nodes = list(self.graph.nodes.values())
        
        if not nodes:
            return {
                'total_count': 0,
                'by_type': {},
                'importance_distribution': {},
                'access_patterns': {},
                'creation_timeline': {}
            }
        
        # Count by type
        type_counts = Counter(node.type.value for node in nodes)
        
        # Importance distribution
        importance_scores = [node.importance_score for node in nodes]
        importance_distribution = {
            'mean': sum(importance_scores) / len(importance_scores),
            'min': min(importance_scores),
            'max': max(importance_scores),
            'high_importance_count': sum(1 for score in importance_scores if score > 0.8),
            'low_importance_count': sum(1 for score in importance_scores if score < 0.2)
        }
        
        # Access patterns
        access_counts = [node.access_count for node in nodes]
        access_patterns = {
            'total_accesses': sum(access_counts),
            'average_accesses': sum(access_counts) / len(access_counts),
            'most_accessed': max(access_counts) if access_counts else 0,
            'never_accessed': sum(1 for count in access_counts if count == 0)
        }
        
        # Creation timeline (last 24 hours)
        now = datetime.now()
        recent_nodes = [
            node for node in nodes 
            if (now - node.created_at).total_seconds() < 86400
        ]
        
        creation_timeline = {
            'created_last_hour': len([
                node for node in recent_nodes 
                if (now - node.created_at).total_seconds() < 3600
            ]),
            'created_last_day': len(recent_nodes),
            'creation_rate_per_hour': len(recent_nodes) / 24 if recent_nodes else 0
        }
        
        return {
            'total_count': len(nodes),
            'by_type': dict(type_counts),
            'importance_distribution': importance_distribution,
            'access_patterns': access_patterns,
            'creation_timeline': creation_timeline
        }
    
    def _collect_edge_stats(self) -> Dict[str, Any]:
        """Collect edge-related statistics"""
        edges = list(self.graph.edges.values())
        
        if not edges:
            return {
                'total_count': 0,
                'by_type': {},
                'weight_distribution': {},
                'confidence_distribution': {}
            }
        
        # Count by type
        type_counts = Counter(edge.type.value for edge in edges)
        
        # Weight distribution
        weights = [edge.weight for edge in edges]
        weight_distribution = {
            'mean': sum(weights) / len(weights),
            'min': min(weights),
            'max': max(weights),
            'high_weight_count': sum(1 for w in weights if w > 0.8),
            'low_weight_count': sum(1 for w in weights if w < 0.2)
        }
        
        # Confidence distribution
        confidences = [edge.confidence for edge in edges]
        confidence_distribution = {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.5)
        }
        
        return {
            'total_count': len(edges),
            'by_type': dict(type_counts),
            'weight_distribution': weight_distribution,
            'confidence_distribution': confidence_distribution
        }
    
    def _collect_connectivity_stats(self) -> Dict[str, Any]:
        """Collect connectivity-related statistics"""
        try:
            import networkx as nx
            
            # Basic connectivity metrics
            if self.graph.graph.number_of_nodes() == 0:
                return {
                    'density': 0.0,
                    'connected_components': 0,
                    'largest_component_size': 0,
                    'isolated_nodes': 0,
                    'average_degree': 0.0,
                    'clustering_coefficient': 0.0
                }
            
            density = nx.density(self.graph.graph)
            connected_components = nx.number_weakly_connected_components(self.graph.graph)
            
            # Largest component
            if connected_components > 0:
                largest_cc = max(nx.weakly_connected_components(self.graph.graph), key=len)
                largest_component_size = len(largest_cc)
            else:
                largest_component_size = 0
            
            # Isolated nodes
            isolated_nodes = len([
                node for node in self.graph.graph.nodes() 
                if self.graph.graph.degree(node) == 0
            ])
            
            # Average degree
            degrees = dict(self.graph.graph.degree())
            average_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0
            
            # Clustering coefficient (for undirected version)
            try:
                undirected = self.graph.graph.to_undirected()
                clustering_coefficient = nx.average_clustering(undirected)
            except:
                clustering_coefficient = 0.0
            
            return {
                'density': density,
                'connected_components': connected_components,
                'largest_component_size': largest_component_size,
                'isolated_nodes': isolated_nodes,
                'average_degree': average_degree,
                'clustering_coefficient': clustering_coefficient
            }
            
        except Exception as e:
            logger.error(f"Error collecting connectivity stats: {e}")
            return {
                'error': str(e),
                'density': 0.0,
                'connected_components': 0,
                'largest_component_size': 0,
                'isolated_nodes': 0,
                'average_degree': 0.0
            }
    
    def _collect_performance_stats(self) -> Dict[str, Any]:
        """Collect performance-related statistics"""
        # Calculate average operation times
        avg_operation_times = {}
        for operation, times in self.operation_times.items():
            if times:
                avg_operation_times[operation] = {
                    'average_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'count': len(times)
                }
        
        # Memory usage estimation
        memory_stats = {
            'estimated_nodes_memory_mb': len(self.graph.nodes) * 0.001,  # Rough estimate
            'estimated_edges_memory_mb': len(self.graph.edges) * 0.0005,
            'cache_sizes': {
                'retrieval_cache': getattr(self.graph, 'retrieval_cache_size', 0),
                'planning_cache': getattr(self.graph, 'planning_cache_size', 0)
            }
        }
        
        return {
            'operation_times': avg_operation_times,
            'error_counts': dict(self.error_counts),
            'memory_stats': memory_stats
        }
    
    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect graph health metrics"""
        health_score = 1.0
        issues = []
        
        # Check node count
        node_count = len(self.graph.nodes)
        if node_count > self.alert_thresholds['max_nodes']:
            health_score -= 0.2
            issues.append(f"High node count: {node_count}")
        
        # Check edge count
        edge_count = len(self.graph.edges)
        if edge_count > self.alert_thresholds['max_edges']:
            health_score -= 0.2
            issues.append(f"High edge count: {edge_count}")
        
        # Check connectivity
        try:
            import networkx as nx
            if self.graph.graph.number_of_nodes() > 0:
                density = nx.density(self.graph.graph)
                if density < self.alert_thresholds['min_connectivity']:
                    health_score -= 0.3
                    issues.append(f"Low connectivity: {density:.3f}")
        except:
            pass
        
        # Check for isolated nodes
        isolated_count = len([
            node for node in self.graph.graph.nodes() 
            if self.graph.graph.degree(node) == 0
        ])
        if isolated_count > self.alert_thresholds['max_isolated_nodes']:
            health_score -= 0.2
            issues.append(f"Too many isolated nodes: {isolated_count}")
        
        # Check error rates
        total_errors = sum(self.error_counts.values())
        if total_errors > 10:
            health_score -= 0.1
            issues.append(f"High error count: {total_errors}")
        
        health_score = max(0.0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'warning' if health_score > 0.5 else 'critical',
            'issues': issues,
            'last_check': datetime.now().isoformat()
        }
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends over the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_stats = [
            stats for stats in self.stats_history 
            if stats.timestamp >= cutoff_time
        ]
        
        if len(recent_stats) < 2:
            return {
                'insufficient_data': True,
                'available_points': len(recent_stats)
            }
        
        # Calculate trends
        trends = {}
        
        # Node count trend
        node_counts = [stats.node_stats['total_count'] for stats in recent_stats]
        trends['node_growth'] = {
            'start_count': node_counts[0],
            'end_count': node_counts[-1],
            'growth': node_counts[-1] - node_counts[0],
            'growth_rate': (node_counts[-1] - node_counts[0]) / max(node_counts[0], 1)
        }
        
        # Edge count trend
        edge_counts = [stats.edge_stats['total_count'] for stats in recent_stats]
        trends['edge_growth'] = {
            'start_count': edge_counts[0],
            'end_count': edge_counts[-1],
            'growth': edge_counts[-1] - edge_counts[0],
            'growth_rate': (edge_counts[-1] - edge_counts[0]) / max(edge_counts[0], 1)
        }
        
        # Health trend
        health_scores = [stats.health_metrics['health_score'] for stats in recent_stats]
        trends['health_trend'] = {
            'start_score': health_scores[0],
            'end_score': health_scores[-1],
            'change': health_scores[-1] - health_scores[0],
            'average': sum(health_scores) / len(health_scores)
        }
        
        return {
            'time_period_hours': hours,
            'data_points': len(recent_stats),
            'trends': trends,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report"""
        current_stats = self.collect_comprehensive_stats()
        trend_analysis = self.get_trend_analysis(24)
        
        # Generate recommendations
        recommendations = []
        
        # Node-based recommendations
        if current_stats.node_stats['total_count'] > 5000:
            recommendations.append("Consider implementing node cleanup policies for old, unused nodes")
        
        if current_stats.node_stats['access_patterns']['never_accessed'] > 100:
            recommendations.append("Many nodes are never accessed - consider relevance-based pruning")
        
        # Edge-based recommendations
        if current_stats.edge_stats['total_count'] > 20000:
            recommendations.append("High edge count detected - monitor performance impact")
        
        # Connectivity recommendations
        if current_stats.connectivity_stats['isolated_nodes'] > 50:
            recommendations.append("High number of isolated nodes - improve graph connectivity")
        
        # Performance recommendations
        if current_stats.performance_stats.get('error_counts'):
            total_errors = sum(current_stats.performance_stats['error_counts'].values())
            if total_errors > 5:
                recommendations.append("Address recurring errors to improve system stability")
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'current_stats': current_stats,
            'trend_analysis': trend_analysis,
            'recommendations': recommendations,
            'overall_health': current_stats.health_metrics['status'],
            'health_score': current_stats.health_metrics['health_score']
        }
    
    def record_operation_time(self, operation: str, time_ms: float):
        """Record the time taken for an operation"""
        self.operation_times[operation].append(time_ms)
        
        # Keep only recent times (last 100 per operation)
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation].pop(0)
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] += 1
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for monitoring dashboard"""
        current_stats = self.collect_comprehensive_stats()
        
        # Format for dashboard display
        dashboard_data = {
            'summary': {
                'total_nodes': current_stats.node_stats['total_count'],
                'total_edges': current_stats.edge_stats['total_count'],
                'health_score': current_stats.health_metrics['health_score'],
                'health_status': current_stats.health_metrics['status']
            },
            'node_distribution': current_stats.node_stats['by_type'],
            'edge_distribution': current_stats.edge_stats['by_type'],
            'connectivity': {
                'density': current_stats.connectivity_stats['density'],
                'components': current_stats.connectivity_stats['connected_components'],
                'isolated_nodes': current_stats.connectivity_stats['isolated_nodes']
            },
            'performance': {
                'avg_operation_times': current_stats.performance_stats.get('operation_times', {}),
                'error_counts': current_stats.performance_stats.get('error_counts', {}),
                'memory_usage': current_stats.performance_stats.get('memory_stats', {})
            },
            'alerts': current_stats.health_metrics.get('issues', []),
            'last_updated': current_stats.timestamp.isoformat()
        }
        
        return dashboard_data
    
    def export_stats_history(self, format: str = 'json') -> str:
        """Export statistics history"""
        if format.lower() == 'json':
            # Convert stats to JSON-serializable format
            serializable_history = []
            for stats in self.stats_history:
                serializable_stats = {
                    'timestamp': stats.timestamp.isoformat(),
                    'node_stats': stats.node_stats,
                    'edge_stats': stats.edge_stats,
                    'connectivity_stats': stats.connectivity_stats,
                    'performance_stats': stats.performance_stats,
                    'health_metrics': stats.health_metrics
                }
                serializable_history.append(serializable_stats)
            
            return json.dumps(serializable_history, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_history(self):
        """Clear statistics history"""
        cleared_count = len(self.stats_history)
        self.stats_history.clear()
        self.operation_times.clear()
        self.error_counts.clear()
        
        logger.info(f"Cleared monitoring history ({cleared_count} entries)")
    
    def set_alert_thresholds(self, thresholds: Dict[str, Any]):
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        logger.info(f"Updated alert thresholds: {thresholds}")
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get current monitoring configuration"""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'alert_thresholds': self.alert_thresholds,
            'max_history_size': self.max_history_size,
            'current_history_size': len(self.stats_history)
        }