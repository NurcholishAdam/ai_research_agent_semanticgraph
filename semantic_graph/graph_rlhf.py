# -*- coding: utf-8 -*-
"""
Graph-Enhanced RLHF Integration
Records user preferences as typed edges and detects reward hacking patterns
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict, Counter

from .graph_core import SemanticGraph, GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    """Types of user preferences"""
    STYLE = "style"
    QUALITY = "quality"
    CONTENT = "content"
    FORMAT = "format"
    TONE = "tone"
    LENGTH = "length"
    COMPLEXITY = "complexity"

@dataclass
class PreferenceRecord:
    """Records a user preference"""
    user_id: str
    preference_type: PreferenceType
    preferred_value: str
    rejected_value: str = ""
    confidence: float = 1.0
    context: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PreferenceGraph:
    """Manages preference relationships in the semantic graph"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.preference_patterns = defaultdict(list)
        self.reward_hacking_indicators = []
        
    def record_preference(self, preference: PreferenceRecord) -> str:
        """Record a user preference in the graph"""
        
        # Create user node if it doesn't exist
        user_node_id = f"user_{preference.user_id}"
        if not self.graph.get_node(user_node_id):
            user_node = GraphNode(
                id=user_node_id,
                type=NodeType.ENTITY,
                label=f"User {preference.user_id}",
                properties={
                    'user_id': preference.user_id,
                    'entity_type': 'user',
                    'created_at': datetime.now().isoformat()
                }
            )
            self.graph.add_node(user_node)
        
        # Create preference node
        pref_node_id = f"pref_{preference.user_id}_{hash(preference.preferred_value) % 10000}"
        pref_node = GraphNode(
            id=pref_node_id,
            type=NodeType.PREFERENCE,
            label=f"{preference.preference_type.value}: {preference.preferred_value}",
            properties={
                'user_id': preference.user_id,
                'preference_type': preference.preference_type.value,
                'preferred_value': preference.preferred_value,
                'rejected_value': preference.rejected_value,
                'confidence': preference.confidence,
                'context': preference.context,
                'timestamp': preference.timestamp.isoformat()
            }
        )
        self.graph.add_node(pref_node)
        
        # Create preference edge
        pref_edge = GraphEdge(
            source_id=user_node_id,
            target_id=pref_node_id,
            type=EdgeType.PREFERRED_STYLE,
            weight=preference.confidence,
            confidence=preference.confidence,
            properties={
                'preference_type': preference.preference_type.value,
                'timestamp': preference.timestamp.isoformat()
            }
        )
        self.graph.add_edge(pref_edge)
        
        # Create rejection edge if applicable
        if preference.rejected_value:
            reject_node_id = f"reject_{preference.user_id}_{hash(preference.rejected_value) % 10000}"
            reject_node = GraphNode(
                id=reject_node_id,
                type=NodeType.STYLE,
                label=f"Rejected: {preference.rejected_value}",
                properties={
                    'style_type': 'rejected',
                    'value': preference.rejected_value,
                    'user_id': preference.user_id
                }
            )
            self.graph.add_node(reject_node)
            
            reject_edge = GraphEdge(
                source_id=user_node_id,
                target_id=reject_node_id,
                type=EdgeType.DOWNVOTED_FOR,
                weight=preference.confidence,
                confidence=preference.confidence
            )
            self.graph.add_edge(reject_edge)
        
        # Update preference patterns
        self.preference_patterns[preference.user_id].append(preference)
        
        # Check for reward hacking patterns
        self._detect_reward_hacking(preference)
        
        logger.debug(f"Recorded preference for user {preference.user_id}: {preference.preference_type.value}")
        return pref_node_id
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user"""
        user_node_id = f"user_{user_id}"
        user_node = self.graph.get_node(user_node_id)
        
        if not user_node:
            return {'preferences': [], 'patterns': {}}
        
        # Get preference nodes connected to user
        preference_neighbors = self.graph.get_neighbors(
            user_node_id,
            edge_types=[EdgeType.PREFERRED_STYLE],
            max_depth=1
        )
        
        preferences = []
        for pref_node in preference_neighbors:
            if pref_node.type == NodeType.PREFERENCE:
                preferences.append({
                    'id': pref_node.id,
                    'type': pref_node.properties.get('preference_type'),
                    'preferred_value': pref_node.properties.get('preferred_value'),
                    'rejected_value': pref_node.properties.get('rejected_value'),
                    'confidence': pref_node.properties.get('confidence'),
                    'timestamp': pref_node.properties.get('timestamp')
                })
        
        # Analyze preference patterns
        patterns = self._analyze_preference_patterns(user_id)
        
        return {
            'user_id': user_id,
            'preferences': preferences,
            'patterns': patterns,
            'total_preferences': len(preferences)
        }
    
    def _analyze_preference_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze patterns in user preferences"""
        user_prefs = self.preference_patterns.get(user_id, [])
        
        if not user_prefs:
            return {}
        
        # Analyze by preference type
        type_counts = Counter(pref.preference_type.value for pref in user_prefs)
        
        # Analyze preference consistency
        consistency_scores = {}
        for pref_type in PreferenceType:
            type_prefs = [p for p in user_prefs if p.preference_type == pref_type]
            if len(type_prefs) > 1:
                # Check consistency of preferred values
                preferred_values = [p.preferred_value for p in type_prefs]
                unique_values = set(preferred_values)
                consistency_scores[pref_type.value] = len(unique_values) / len(preferred_values)
        
        # Detect preference evolution over time
        evolution = self._detect_preference_evolution(user_prefs)
        
        return {
            'type_distribution': dict(type_counts),
            'consistency_scores': consistency_scores,
            'evolution': evolution,
            'most_common_type': type_counts.most_common(1)[0] if type_counts else None
        }
    
    def _detect_preference_evolution(self, preferences: List[PreferenceRecord]) -> Dict[str, Any]:
        """Detect how preferences evolve over time"""
        if len(preferences) < 2:
            return {'evolution_detected': False}
        
        # Sort by timestamp
        sorted_prefs = sorted(preferences, key=lambda x: x.timestamp)
        
        # Group by preference type and analyze changes
        type_evolution = {}
        for pref_type in PreferenceType:
            type_prefs = [p for p in sorted_prefs if p.preference_type == pref_type]
            if len(type_prefs) > 1:
                # Check if preferences changed over time
                values = [p.preferred_value for p in type_prefs]
                changed = len(set(values)) > 1
                
                type_evolution[pref_type.value] = {
                    'changed': changed,
                    'progression': values if changed else None,
                    'stability': 1.0 - (len(set(values)) / len(values))
                }
        
        return {
            'evolution_detected': bool(type_evolution),
            'type_evolution': type_evolution,
            'overall_stability': sum(
                evo.get('stability', 1.0) for evo in type_evolution.values()
            ) / len(type_evolution) if type_evolution else 1.0
        }
    
    def _detect_reward_hacking(self, preference: PreferenceRecord):
        """Detect potential reward hacking patterns"""
        user_id = preference.user_id
        
        # Check for suspicious patterns
        suspicious_indicators = []
        
        # 1. Extremely high confidence scores consistently
        user_prefs = self.preference_patterns.get(user_id, [])
        if len(user_prefs) >= 3:
            recent_confidences = [p.confidence for p in user_prefs[-3:]]
            if all(conf >= 0.95 for conf in recent_confidences):
                suspicious_indicators.append({
                    'type': 'high_confidence_pattern',
                    'description': 'Consistently extremely high confidence scores',
                    'severity': 0.6
                })
        
        # 2. Rapid preference changes
        if len(user_prefs) >= 2:
            last_pref = user_prefs[-2]
            if (last_pref.preference_type == preference.preference_type and
                last_pref.preferred_value != preference.preferred_value):
                time_diff = (preference.timestamp - last_pref.timestamp).total_seconds()
                if time_diff < 300:  # Less than 5 minutes
                    suspicious_indicators.append({
                        'type': 'rapid_preference_change',
                        'description': 'Preference changed very quickly',
                        'severity': 0.7
                    })
        
        # 3. Contradictory preferences
        for existing_pref in user_prefs:
            if (existing_pref.preference_type == preference.preference_type and
                existing_pref.preferred_value == preference.rejected_value):
                suspicious_indicators.append({
                    'type': 'contradictory_preference',
                    'description': 'Current preference contradicts previous preference',
                    'severity': 0.8
                })
        
        # Record suspicious patterns
        if suspicious_indicators:
            self.reward_hacking_indicators.append({
                'user_id': user_id,
                'preference_id': f"pref_{user_id}_{hash(preference.preferred_value) % 10000}",
                'indicators': suspicious_indicators,
                'timestamp': preference.timestamp,
                'overall_suspicion': max(ind['severity'] for ind in suspicious_indicators)
            })
            
            logger.warning(f"Potential reward hacking detected for user {user_id}")

class GraphRLHFIntegration:
    """Main class for integrating RLHF with semantic graph"""
    
    def __init__(self, semantic_graph: SemanticGraph):
        self.graph = semantic_graph
        self.preference_graph = PreferenceGraph(semantic_graph)
        
        # RLHF statistics
        self.rlhf_stats = {
            'preferences_recorded': 0,
            'users_tracked': 0,
            'reward_hacking_detected': 0,
            'last_updated': datetime.now()
        }
        
        logger.info("Graph-enhanced RLHF integration initialized")
    
    def record_feedback(self, user_id: str, preferred_content: str, rejected_content: str = "",
                       feedback_type: str = "quality", confidence: float = 1.0,
                       context: str = "") -> str:
        """Record user feedback as graph preferences"""
        
        # Determine preference type
        pref_type = PreferenceType.QUALITY
        if feedback_type.lower() in ['style', 'tone', 'format', 'length', 'complexity']:
            pref_type = PreferenceType(feedback_type.lower())
        
        # Create preference record
        preference = PreferenceRecord(
            user_id=user_id,
            preference_type=pref_type,
            preferred_value=preferred_content,
            rejected_value=rejected_content,
            confidence=confidence,
            context=context
        )
        
        # Record in preference graph
        pref_node_id = self.preference_graph.record_preference(preference)
        
        # Update statistics
        self.rlhf_stats['preferences_recorded'] += 1
        if user_id not in [p.user_id for p in self.preference_graph.preference_patterns.keys()]:
            self.rlhf_stats['users_tracked'] += 1
        self.rlhf_stats['last_updated'] = datetime.now()
        
        return pref_node_id
    
    def get_generation_guidance(self, user_id: str, content_type: str = "general") -> Dict[str, Any]:
        """Get guidance for content generation based on user preferences"""
        
        user_preferences = self.preference_graph.get_user_preferences(user_id)
        
        if not user_preferences['preferences']:
            return {
                'guidance_available': False,
                'default_guidance': {
                    'style': 'neutral',
                    'quality': 'high',
                    'tone': 'professional'
                }
            }
        
        # Extract guidance from preferences
        guidance = {}
        for pref in user_preferences['preferences']:
            pref_type = pref['type']
            if pref_type and pref['preferred_value']:
                guidance[pref_type] = pref['preferred_value']
        
        # Add pattern-based guidance
        patterns = user_preferences['patterns']
        if patterns.get('most_common_type'):
            most_common_type, count = patterns['most_common_type']
            guidance['primary_focus'] = most_common_type
        
        return {
            'guidance_available': True,
            'user_id': user_id,
            'guidance': guidance,
            'confidence': self._calculate_guidance_confidence(user_preferences),
            'patterns': patterns
        }
    
    def detect_reward_hacking(self, user_id: str = None) -> Dict[str, Any]:
        """Detect reward hacking patterns"""
        
        all_indicators = self.preference_graph.reward_hacking_indicators
        
        if user_id:
            user_indicators = [ind for ind in all_indicators if ind['user_id'] == user_id]
        else:
            user_indicators = all_indicators
        
        # Analyze patterns
        if not user_indicators:
            return {
                'reward_hacking_detected': False,
                'user_id': user_id,
                'indicators': []
            }
        
        # Group by user and analyze
        user_analysis = defaultdict(list)
        for indicator in user_indicators:
            user_analysis[indicator['user_id']].append(indicator)
        
        # Calculate risk scores
        risk_analysis = {}
        for uid, indicators in user_analysis.items():
            avg_suspicion = sum(ind['overall_suspicion'] for ind in indicators) / len(indicators)
            risk_analysis[uid] = {
                'indicator_count': len(indicators),
                'average_suspicion': avg_suspicion,
                'risk_level': 'high' if avg_suspicion > 0.7 else 'medium' if avg_suspicion > 0.4 else 'low',
                'latest_incident': max(indicators, key=lambda x: x['timestamp'])['timestamp']
            }
        
        return {
            'reward_hacking_detected': len(user_indicators) > 0,
            'user_id': user_id,
            'indicators': user_indicators,
            'risk_analysis': risk_analysis,
            'total_suspicious_users': len(user_analysis)
        }
    
    def get_preference_insights(self, user_id: str = None) -> Dict[str, Any]:
        """Get insights about user preferences"""
        
        if user_id:
            # Single user insights
            user_prefs = self.preference_graph.get_user_preferences(user_id)
            return {
                'user_insights': user_prefs,
                'recommendations': self._generate_user_recommendations(user_prefs)
            }
        else:
            # Global insights
            all_users = list(self.preference_graph.preference_patterns.keys())
            
            global_patterns = {
                'total_users': len(all_users),
                'preference_distribution': {},
                'consistency_analysis': {},
                'evolution_trends': {}
            }
            
            # Aggregate patterns across users
            all_type_counts = Counter()
            consistency_scores = []
            
            for uid in all_users:
                user_prefs = self.preference_graph.get_user_preferences(uid)
                patterns = user_prefs['patterns']
                
                # Aggregate type distribution
                if 'type_distribution' in patterns:
                    for ptype, count in patterns['type_distribution'].items():
                        all_type_counts[ptype] += count
                
                # Aggregate consistency scores
                if 'consistency_scores' in patterns:
                    consistency_scores.extend(patterns['consistency_scores'].values())
            
            global_patterns['preference_distribution'] = dict(all_type_counts)
            global_patterns['average_consistency'] = (
                sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            )
            
            return {
                'global_insights': global_patterns,
                'user_count': len(all_users),
                'total_preferences': sum(all_type_counts.values())
            }
    
    def _calculate_guidance_confidence(self, user_preferences: Dict[str, Any]) -> float:
        """Calculate confidence in generation guidance"""
        preferences = user_preferences['preferences']
        patterns = user_preferences['patterns']
        
        if not preferences:
            return 0.0
        
        # Base confidence from number of preferences
        count_confidence = min(1.0, len(preferences) / 10.0)
        
        # Consistency confidence
        consistency_scores = patterns.get('consistency_scores', {})
        avg_consistency = sum(consistency_scores.values()) / len(consistency_scores) if consistency_scores else 0.5
        
        # Recency confidence (prefer recent preferences)
        recent_prefs = [p for p in preferences if 'timestamp' in p]
        if recent_prefs:
            # Simple recency boost
            recency_confidence = 0.8  # Assume recent if we have timestamps
        else:
            recency_confidence = 0.5
        
        # Combine factors
        overall_confidence = (count_confidence * 0.4 + avg_consistency * 0.4 + recency_confidence * 0.2)
        
        return min(1.0, overall_confidence)
    
    def _generate_user_recommendations(self, user_preferences: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving user experience"""
        recommendations = []
        
        patterns = user_preferences['patterns']
        preferences = user_preferences['preferences']
        
        # Recommendation based on consistency
        consistency_scores = patterns.get('consistency_scores', {})
        for ptype, score in consistency_scores.items():
            if score < 0.5:  # Low consistency
                recommendations.append(
                    f"Consider clarifying your {ptype} preferences - they seem inconsistent"
                )
        
        # Recommendation based on preference coverage
        covered_types = set(p['type'] for p in preferences if p['type'])
        all_types = set(ptype.value for ptype in PreferenceType)
        missing_types = all_types - covered_types
        
        if len(missing_types) > 3:
            recommendations.append(
                f"Consider providing preferences for: {', '.join(list(missing_types)[:3])}"
            )
        
        # Recommendation based on evolution
        evolution = patterns.get('evolution', {})
        if evolution.get('evolution_detected') and evolution.get('overall_stability', 1.0) < 0.3:
            recommendations.append(
                "Your preferences seem to change frequently - consider what you truly prefer"
            )
        
        return recommendations
    
    def get_rlhf_statistics(self) -> Dict[str, Any]:
        """Get RLHF integration statistics"""
        reward_hacking_analysis = self.detect_reward_hacking()
        
        return {
            'basic_stats': self.rlhf_stats,
            'reward_hacking': {
                'total_incidents': len(self.preference_graph.reward_hacking_indicators),
                'suspicious_users': reward_hacking_analysis.get('total_suspicious_users', 0)
            },
            'preference_coverage': {
                'total_users': len(self.preference_graph.preference_patterns),
                'total_preferences': sum(
                    len(prefs) for prefs in self.preference_graph.preference_patterns.values()
                )
            },
            'graph_integration': {
                'preference_nodes': len([
                    n for n in self.graph.nodes.values() 
                    if n.type == NodeType.PREFERENCE
                ]),
                'preference_edges': len([
                    e for e in self.graph.edges.values() 
                    if e.type in [EdgeType.PREFERRED_STYLE, EdgeType.DOWNVOTED_FOR]
                ])
            }
        }