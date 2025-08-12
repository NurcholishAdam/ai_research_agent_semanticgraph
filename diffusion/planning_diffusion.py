# -*- coding: utf-8 -*-
"""
Planning and Reflection with Generative Sampling
Uses diffusion models for diverse plan generation and trajectory refinement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import logging
from collections import Counter

from .diffusion_core import DiffusionCore, DiffusionConfig

logger = logging.getLogger(__name__)

@dataclass
class PlanningConfig:
    """Configuration for planning diffusion"""
    num_plan_candidates: int = 5
    max_plan_steps: int = 10
    diversity_weight: float = 0.3
    quality_weight: float = 0.7
    refinement_iterations: int = 3
    trajectory_noise_level: float = 0.15
    voting_threshold: float = 0.6
    creativity_boost: float = 0.2
    
class PlanningDiffusionModel(nn.Module):
    """Specialized diffusion model for planning tasks"""
    
    def __init__(self, config: DiffusionConfig, planning_config: PlanningConfig):
        super().__init__()
        self.config = config
        self.planning_config = planning_config
        
        # Planning-specific layers
        self.plan_encoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 2),
            nn.LayerNorm(config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.model_dim)
        )
        
        # Step sequence modeling
        self.step_attention = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Task decomposition head
        self.decomposition_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, planning_config.max_plan_steps)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for planning diffusion"""
        # Encode planning context
        plan_features = self.plan_encoder(x)
        
        # Apply self-attention for step dependencies
        attended_features, _ = self.step_attention(
            plan_features.unsqueeze(1), 
            plan_features.unsqueeze(1), 
            plan_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Combine with task context if provided
        if task_context is not None:
            attended_features = attended_features + task_context
            
        return attended_features

class DiversePlanProposer:
    """Generates diverse plan proposals using diffusion sampling"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: PlanningConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or PlanningConfig()
        self.planning_model = PlanningDiffusionModel(diffusion_core.config, self.config)
        
    def generate_plan_candidates(self, research_question: str, 
                                context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate multiple diverse plan candidates"""
        logger.info(f"Generating {self.config.num_plan_candidates} diverse plan candidates")
        
        # Encode research question
        question_embedding = self.diffusion_core.encode_text([research_question])
        
        # Encode context if provided
        context_embedding = None
        if context:
            context_text = self._context_to_text(context)
            context_embedding = self.diffusion_core.encode_text([context_text])
            
        plan_candidates = []
        
        for i in range(self.config.num_plan_candidates):
            try:
                # Sample different denoising trajectory
                plan_embedding = self._sample_plan_trajectory(
                    question_embedding, context_embedding, trajectory_id=i
                )
                
                # Convert embedding to plan structure
                plan_structure = self._embedding_to_plan(plan_embedding, research_question)
                
                # Add diversity and quality scores
                plan_structure.update({
                    'candidate_id': i,
                    'diversity_score': self._calculate_diversity_score(plan_structure, plan_candidates),
                    'estimated_quality': self._estimate_plan_quality(plan_structure, research_question)
                })
                
                plan_candidates.append(plan_structure)
                
            except Exception as e:
                logger.warning(f"Failed to generate plan candidate {i}: {e}")
                continue
                
        logger.info(f"Successfully generated {len(plan_candidates)} plan candidates")
        return plan_candidates
    
    def _sample_plan_trajectory(self, question_embedding: torch.Tensor, 
                               context_embedding: Optional[torch.Tensor] = None,
                               trajectory_id: int = 0) -> torch.Tensor:
        """Sample a specific denoising trajectory for plan generation"""
        # Create unique sampling path for diversity
        torch.manual_seed(42 + trajectory_id)  # Ensure reproducible diversity
        
        # Start with noise
        shape = question_embedding.shape
        noisy_plan = torch.randn(shape, device=self.diffusion_core.device)
        
        # Add creativity boost for more diverse trajectories
        creativity_noise = torch.randn_like(noisy_plan) * self.config.creativity_boost
        noisy_plan = noisy_plan + creativity_noise
        
        # Custom denoising trajectory
        num_steps = self.diffusion_core.config.num_timesteps // 2  # Faster sampling
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, 
                                 device=self.diffusion_core.device)
        
        current_plan = noisy_plan
        for t in timesteps:
            t_batch = t.repeat(shape[0])
            
            # Condition on question and context
            condition = question_embedding
            if context_embedding is not None:
                condition = torch.cat([question_embedding, context_embedding], dim=-1)
                
            current_plan = self.diffusion_core.denoise_step(current_plan, t_batch, condition)
            
        return current_plan
    
    def _embedding_to_plan(self, plan_embedding: torch.Tensor, research_question: str) -> Dict[str, Any]:
        """Convert plan embedding to structured plan"""
        # This is a simplified conversion - in practice, you'd use a trained decoder
        
        # Generate plan steps based on embedding characteristics
        embedding_np = plan_embedding.cpu().numpy().flatten()
        
        # Use embedding values to determine plan characteristics
        num_steps = min(max(3, int(abs(embedding_np[0]) * 10)), self.config.max_plan_steps)
        
        # Generate diverse plan steps
        plan_templates = [
            "Search for recent research on {topic}",
            "Analyze existing literature about {topic}",
            "Identify key experts and organizations in {topic}",
            "Examine current challenges in {topic}",
            "Investigate practical applications of {topic}",
            "Review historical development of {topic}",
            "Compare different approaches to {topic}",
            "Synthesize findings about {topic}",
            "Evaluate evidence quality for {topic}",
            "Generate comprehensive summary of {topic}"
        ]
        
        # Select steps based on embedding values
        selected_indices = []
        for i in range(num_steps):
            idx = int(abs(embedding_np[i % len(embedding_np)]) * len(plan_templates)) % len(plan_templates)
            if idx not in selected_indices:
                selected_indices.append(idx)
            else:
                # Find next available index
                for j in range(len(plan_templates)):
                    if j not in selected_indices:
                        selected_indices.append(j)
                        break
        
        # Create plan steps
        topic = self._extract_topic_from_question(research_question)
        plan_steps = []
        for i, idx in enumerate(selected_indices[:num_steps]):
            step = {
                'step_id': i + 1,
                'description': plan_templates[idx].format(topic=topic),
                'estimated_effort': abs(embedding_np[(i + 1) % len(embedding_np)]),
                'dependencies': [] if i == 0 else [i],  # Simple dependency chain
                'tools_suggested': self._suggest_tools_for_step(plan_templates[idx])
            }
            plan_steps.append(step)
            
        return {
            'research_question': research_question,
            'plan_steps': plan_steps,
            'total_steps': len(plan_steps),
            'estimated_complexity': np.mean([step['estimated_effort'] for step in plan_steps]),
            'generation_method': 'diffusion_planning'
        }
    
    def _extract_topic_from_question(self, question: str) -> str:
        """Extract main topic from research question"""
        # Simple topic extraction - in practice, use NLP techniques
        words = question.lower().split()
        
        # Remove common question words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        topic_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        return ' '.join(topic_words[:3]) if topic_words else 'the topic'
    
    def _suggest_tools_for_step(self, step_template: str) -> List[str]:
        """Suggest appropriate tools for a plan step"""
        tool_mapping = {
            'search': ['web_search', 'academic_search'],
            'analyze': ['document_processor', 'data_visualization'],
            'identify': ['web_search', 'expert_finder'],
            'examine': ['document_processor', 'web_research'],
            'investigate': ['web_search', 'data_analysis'],
            'review': ['document_processor', 'timeline_analysis'],
            'compare': ['data_visualization', 'comparison_tools'],
            'synthesize': ['summary_generator', 'knowledge_graph'],
            'evaluate': ['fact_checker', 'quality_assessor'],
            'generate': ['report_generator', 'summary_tools']
        }
        
        suggested_tools = []
        for keyword, tools in tool_mapping.items():
            if keyword in step_template.lower():
                suggested_tools.extend(tools)
                
        return list(set(suggested_tools)) if suggested_tools else ['web_search']
    
    def _calculate_diversity_score(self, plan: Dict[str, Any], existing_plans: List[Dict[str, Any]]) -> float:
        """Calculate diversity score compared to existing plans"""
        if not existing_plans:
            return 1.0
            
        # Compare plan steps
        current_steps = set(step['description'] for step in plan['plan_steps'])
        
        diversity_scores = []
        for existing_plan in existing_plans:
            existing_steps = set(step['description'] for step in existing_plan['plan_steps'])
            
            # Jaccard similarity
            intersection = len(current_steps.intersection(existing_steps))
            union = len(current_steps.union(existing_steps))
            similarity = intersection / union if union > 0 else 0
            diversity = 1 - similarity
            diversity_scores.append(diversity)
            
        return np.mean(diversity_scores)
    
    def _estimate_plan_quality(self, plan: Dict[str, Any], research_question: str) -> float:
        """Estimate plan quality based on various factors"""
        quality_factors = []
        
        # Step count appropriateness
        num_steps = len(plan['plan_steps'])
        step_quality = 1.0 - abs(num_steps - 6) / 10  # Prefer around 6 steps
        quality_factors.append(max(0, step_quality))
        
        # Complexity appropriateness
        complexity = plan.get('estimated_complexity', 0.5)
        complexity_quality = 1.0 - abs(complexity - 0.5)  # Prefer moderate complexity
        quality_factors.append(complexity_quality)
        
        # Tool diversity
        all_tools = []
        for step in plan['plan_steps']:
            all_tools.extend(step.get('tools_suggested', []))
        tool_diversity = len(set(all_tools)) / max(len(all_tools), 1)
        quality_factors.append(tool_diversity)
        
        return np.mean(quality_factors)
    
    def _context_to_text(self, context: Dict[str, Any]) -> str:
        """Convert context dictionary to text"""
        context_parts = []
        
        if 'previous_findings' in context:
            context_parts.append(f"Previous findings: {context['previous_findings']}")
            
        if 'available_tools' in context:
            tools = ', '.join(context['available_tools'])
            context_parts.append(f"Available tools: {tools}")
            
        if 'time_constraints' in context:
            context_parts.append(f"Time constraints: {context['time_constraints']}")
            
        return '. '.join(context_parts)

class PlanVotingSystem:
    """Voting system for selecting best plans from candidates"""
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        
    def vote_on_plans(self, plan_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vote on plan candidates to select the best one"""
        logger.info(f"Voting on {len(plan_candidates)} plan candidates")
        
        if not plan_candidates:
            raise ValueError("No plan candidates provided for voting")
            
        # Calculate composite scores
        scored_plans = []
        for plan in plan_candidates:
            composite_score = (
                self.config.diversity_weight * plan.get('diversity_score', 0) +
                self.config.quality_weight * plan.get('estimated_quality', 0)
            )
            
            scored_plan = plan.copy()
            scored_plan['composite_score'] = composite_score
            scored_plans.append(scored_plan)
            
        # Sort by composite score
        scored_plans.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Select winner
        winner = scored_plans[0]
        
        # Calculate voting confidence
        if len(scored_plans) > 1:
            score_gap = winner['composite_score'] - scored_plans[1]['composite_score']
            confidence = min(1.0, score_gap / 0.5)  # Normalize confidence
        else:
            confidence = 1.0
            
        return {
            'selected_plan': winner,
            'all_candidates': scored_plans,
            'voting_confidence': confidence,
            'selection_criteria': {
                'diversity_weight': self.config.diversity_weight,
                'quality_weight': self.config.quality_weight
            }
        }

class TrajectoryRefiner:
    """Refines failed agent trajectories using reverse diffusion"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: PlanningConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or PlanningConfig()
        
    def refine_failed_trajectory(self, failed_trajectory: List[Dict[str, Any]], 
                               failure_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine a failed trajectory using reverse diffusion"""
        logger.info(f"Refining failed trajectory with {len(failed_trajectory)} steps")
        
        if not failed_trajectory:
            return []
            
        # Convert trajectory to embeddings
        trajectory_embeddings = self._trajectory_to_embeddings(failed_trajectory)
        
        # Add noise based on failure severity
        failure_severity = failure_context.get('severity', 0.5)
        noise_level = self.config.trajectory_noise_level * failure_severity
        
        refined_embeddings = []
        for embedding in trajectory_embeddings:
            # Add controlled noise
            noise = torch.randn_like(embedding) * noise_level
            noisy_embedding = embedding + noise
            
            # Apply reverse diffusion iterations
            refined_embedding = self._apply_refinement_iterations(
                noisy_embedding, embedding, failure_context
            )
            refined_embeddings.append(refined_embedding)
            
        # Convert back to trajectory
        refined_trajectory = self._embeddings_to_trajectory(
            refined_embeddings, failed_trajectory, failure_context
        )
        
        logger.info(f"Refined trajectory has {len(refined_trajectory)} steps")
        return refined_trajectory
    
    def _trajectory_to_embeddings(self, trajectory: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Convert trajectory steps to embeddings"""
        embeddings = []
        
        for step in trajectory:
            # Create text representation of step
            step_text = self._step_to_text(step)
            
            # Encode to embedding
            embedding = self.diffusion_core.encode_text([step_text])
            embeddings.append(embedding.squeeze(0))
            
        return embeddings
    
    def _step_to_text(self, step: Dict[str, Any]) -> str:
        """Convert trajectory step to text representation"""
        parts = []
        
        if 'action' in step:
            parts.append(f"Action: {step['action']}")
            
        if 'observation' in step:
            parts.append(f"Observation: {step['observation']}")
            
        if 'reasoning' in step:
            parts.append(f"Reasoning: {step['reasoning']}")
            
        if 'result' in step:
            parts.append(f"Result: {step['result']}")
            
        return '. '.join(parts) if parts else "Empty step"
    
    def _apply_refinement_iterations(self, noisy_embedding: torch.Tensor, 
                                   original_embedding: torch.Tensor,
                                   failure_context: Dict[str, Any]) -> torch.Tensor:
        """Apply reverse diffusion iterations for refinement"""
        current_embedding = noisy_embedding
        
        for iteration in range(self.config.refinement_iterations):
            # Calculate timestep for this iteration
            t = torch.tensor([int((self.config.refinement_iterations - iteration - 1) * 
                                self.diffusion_core.config.num_timesteps / self.config.refinement_iterations)], 
                           device=self.diffusion_core.device)
            t_batch = t.repeat(1)
            
            # Use original embedding as condition for guidance
            current_embedding = self.diffusion_core.denoise_step(
                current_embedding.unsqueeze(0), t_batch, original_embedding.unsqueeze(0)
            ).squeeze(0)
            
        return current_embedding
    
    def _embeddings_to_trajectory(self, embeddings: List[torch.Tensor], 
                                 original_trajectory: List[Dict[str, Any]],
                                 failure_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert refined embeddings back to trajectory"""
        refined_trajectory = []
        
        for i, (embedding, original_step) in enumerate(zip(embeddings, original_trajectory)):
            # Create refined step based on embedding and original step
            refined_step = original_step.copy()
            
            # Modify step based on failure context
            if 'error_type' in failure_context:
                refined_step = self._apply_error_specific_fixes(refined_step, failure_context)
                
            # Add refinement metadata
            refined_step.update({
                'refined': True,
                'refinement_iteration': i,
                'original_step_id': original_step.get('step_id', i)
            })
            
            refined_trajectory.append(refined_step)
            
        return refined_trajectory
    
    def _apply_error_specific_fixes(self, step: Dict[str, Any], 
                                  failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific fixes based on error type"""
        error_type = failure_context.get('error_type', 'unknown')
        
        if error_type == 'tool_failure':
            # Suggest alternative tools
            if 'tools_suggested' in step:
                alternative_tools = ['web_search', 'document_processor', 'data_analysis']
                step['tools_suggested'] = alternative_tools
                
        elif error_type == 'insufficient_information':
            # Add more comprehensive search steps
            if 'action' in step:
                step['action'] = f"Comprehensive {step['action']}"
                
        elif error_type == 'reasoning_error':
            # Add verification steps
            step['verification_required'] = True
            
        return step

class PlanningDiffusion:
    """Main planning diffusion system integrating all components"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: PlanningConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or PlanningConfig()
        
        # Initialize components
        self.plan_proposer = DiversePlanProposer(diffusion_core, config)
        self.voting_system = PlanVotingSystem(config)
        self.trajectory_refiner = TrajectoryRefiner(diffusion_core, config)
        
    def generate_research_plan(self, research_question: str, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive research plan using diffusion sampling"""
        logger.info(f"Generating research plan for: {research_question}")
        
        # Generate diverse plan candidates
        plan_candidates = self.plan_proposer.generate_plan_candidates(research_question, context)
        
        if not plan_candidates:
            raise ValueError("Failed to generate any plan candidates")
            
        # Vote on best plan
        voting_result = self.voting_system.vote_on_plans(plan_candidates)
        
        # Prepare final result
        result = {
            'research_question': research_question,
            'selected_plan': voting_result['selected_plan'],
            'alternative_plans': voting_result['all_candidates'][1:],  # Exclude winner
            'planning_confidence': voting_result['voting_confidence'],
            'generation_stats': {
                'num_candidates_generated': len(plan_candidates),
                'diversity_range': self._calculate_diversity_range(plan_candidates),
                'quality_range': self._calculate_quality_range(plan_candidates)
            }
        }
        
        logger.info(f"Selected plan with {len(result['selected_plan']['plan_steps'])} steps")
        return result
    
    def refine_execution_trajectory(self, trajectory: List[Dict[str, Any]], 
                                  failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Refine failed execution trajectory"""
        logger.info("Refining failed execution trajectory")
        
        refined_trajectory = self.trajectory_refiner.refine_failed_trajectory(trajectory, failure_info)
        
        return {
            'original_trajectory': trajectory,
            'refined_trajectory': refined_trajectory,
            'refinement_applied': True,
            'failure_context': failure_info,
            'improvement_suggestions': self._generate_improvement_suggestions(trajectory, refined_trajectory)
        }
    
    def _calculate_diversity_range(self, plans: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate diversity score range"""
        diversity_scores = [plan.get('diversity_score', 0) for plan in plans]
        return min(diversity_scores), max(diversity_scores)
    
    def _calculate_quality_range(self, plans: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate quality score range"""
        quality_scores = [plan.get('estimated_quality', 0) for plan in plans]
        return min(quality_scores), max(quality_scores)
    
    def _generate_improvement_suggestions(self, original: List[Dict[str, Any]], 
                                        refined: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions based on refinement"""
        suggestions = []
        
        if len(refined) != len(original):
            suggestions.append(f"Trajectory length changed from {len(original)} to {len(refined)} steps")
            
        # Compare step types
        original_actions = [step.get('action', '') for step in original]
        refined_actions = [step.get('action', '') for step in refined]
        
        if original_actions != refined_actions:
            suggestions.append("Action sequence was modified for better execution")
            
        suggestions.append("Applied diffusion-based trajectory refinement")
        
        return suggestions