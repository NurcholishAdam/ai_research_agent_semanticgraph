# -*- coding: utf-8 -*-
"""
Research Agent Integration with Diffusion Models
Integrates all diffusion capabilities into the main research agent
"""

import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .diffusion_core import DiffusionCore, DiffusionConfig
from .synthetic_data_generator import SyntheticDataGenerator, DataAugmentationConfig
from .denoising_layer import DenoisingLayer, NoiseConfig
from .planning_diffusion import PlanningDiffusion, PlanningConfig
from .vision_creativity_agents import ImageDrivenWebAgent, IdeaExplorationAgent, VisionConfig
from .rlhf_diffusion_integration import RLHFDiffusionIntegrator, AlignmentConfig

logger = logging.getLogger(__name__)

@dataclass
class DiffusionIntegrationConfig:
    """Configuration for diffusion integration"""
    enable_synthetic_data: bool = True
    enable_denoising: bool = True
    enable_planning_diffusion: bool = True
    enable_vision_creativity: bool = True
    enable_rlhf_integration: bool = True
    
    # Component-specific configs
    diffusion_config: DiffusionConfig = None
    augmentation_config: DataAugmentationConfig = None
    noise_config: NoiseConfig = None
    planning_config: PlanningConfig = None
    vision_config: VisionConfig = None
    alignment_config: AlignmentConfig = None
    
    def __post_init__(self):
        if self.diffusion_config is None:
            self.diffusion_config = DiffusionConfig()
        if self.augmentation_config is None:
            self.augmentation_config = DataAugmentationConfig()
        if self.noise_config is None:
            self.noise_config = NoiseConfig()
        if self.planning_config is None:
            self.planning_config = PlanningConfig()
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.alignment_config is None:
            self.alignment_config = AlignmentConfig()

class DiffusionEnhancedResearchAgent:
    """Research agent enhanced with comprehensive diffusion capabilities"""
    
    def __init__(self, config: DiffusionIntegrationConfig = None):
        self.config = config or DiffusionIntegrationConfig()
        
        # Initialize core diffusion system
        self.diffusion_core = DiffusionCore(self.config.diffusion_config)
        
        # Initialize components based on configuration
        self.synthetic_data_generator = None
        self.denoising_layer = None
        self.planning_diffusion = None
        self.image_driven_web_agent = None
        self.idea_exploration_agent = None
        self.rlhf_integrator = None
        
        self._initialize_components()
        
        # Integration state
        self.is_initialized = False
        self.training_contexts = []
        
    def _initialize_components(self):
        """Initialize diffusion components based on configuration"""
        logger.info("Initializing diffusion-enhanced research agent components")
        
        if self.config.enable_synthetic_data:
            self.synthetic_data_generator = SyntheticDataGenerator(
                self.config.diffusion_config, 
                self.config.augmentation_config
            )
            logger.debug("Synthetic data generator initialized")
            
        if self.config.enable_denoising:
            self.denoising_layer = DenoisingLayer(
                self.diffusion_core, 
                self.config.noise_config
            )
            logger.debug("Denoising layer initialized")
            
        if self.config.enable_planning_diffusion:
            self.planning_diffusion = PlanningDiffusion(
                self.diffusion_core, 
                self.config.planning_config
            )
            logger.debug("Planning diffusion initialized")
            
        if self.config.enable_vision_creativity:
            self.image_driven_web_agent = ImageDrivenWebAgent(
                self.diffusion_core, 
                self.config.vision_config
            )
            self.idea_exploration_agent = IdeaExplorationAgent(
                self.diffusion_core, 
                self.config.vision_config
            )
            logger.debug("Vision and creativity agents initialized")
            
        if self.config.enable_rlhf_integration:
            self.rlhf_integrator = RLHFDiffusionIntegrator(
                self.diffusion_core, 
                self.config.alignment_config
            )
            logger.debug("RLHF integration initialized")
            
        logger.info("All diffusion components initialized successfully")
    
    def initialize_with_training_data(self, training_contexts: List[str], 
                                    rlhf_data: List[Dict[str, Any]] = None):
        """Initialize the diffusion models with training data"""
        logger.info(f"Initializing diffusion models with {len(training_contexts)} training contexts")
        
        self.training_contexts = training_contexts
        
        # Train core diffusion model
        self.diffusion_core.train(training_contexts)
        
        # Initialize synthetic data generator
        if self.synthetic_data_generator:
            self.synthetic_data_generator.initialize(training_contexts)
            
        # Initialize RLHF integration if data provided
        if self.rlhf_integrator and rlhf_data:
            self.rlhf_integrator.enhance_rlhf_training(rlhf_data)
            
        self.is_initialized = True
        logger.info("Diffusion models initialization completed")
    
    # Stage 1: Synthetic Data Generation
    def augment_memory_contexts(self, contexts: List[str], 
                              metadata: List[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Augment memory contexts with synthetic paraphrases"""
        if not self.synthetic_data_generator:
            logger.warning("Synthetic data generator not enabled")
            return contexts, metadata or [{}] * len(contexts)
            
        return self.synthetic_data_generator.augment_langmem_contexts(contexts, metadata)
    
    def generate_multimodal_content(self, csv_data) -> List[Dict[str, Any]]:
        """Generate synthetic charts and diagrams from CSV data"""
        if not self.synthetic_data_generator:
            logger.warning("Synthetic data generator not enabled")
            return []
            
        return self.synthetic_data_generator.generate_multimodal_content(csv_data)
    
    def balance_training_dataset(self, training_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance training dataset by generating synthetic examples"""
        if not self.synthetic_data_generator:
            logger.warning("Synthetic data generator not enabled")
            return training_examples
            
        return self.synthetic_data_generator.balance_training_dataset(training_examples)
    
    # Stage 2: Denoising Layer
    def robust_retrieval_query(self, query_embedding: torch.Tensor, 
                             candidate_embeddings: torch.Tensor,
                             top_k: int = 5) -> Dict[str, Any]:
        """Process retrieval query with noise robustness"""
        if not self.denoising_layer:
            logger.warning("Denoising layer not enabled")
            # Fallback to simple cosine similarity
            similarities = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                candidate_embeddings, 
                dim=1
            )
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(similarities)))
            return {
                'retrieved_indices': top_k_indices.tolist(),
                'similarities': top_k_values.tolist(),
                'summary_embedding': query_embedding,
                'coherence_score': 1.0,
                'denoised': False
            }
            
        return self.denoising_layer.process_retrieval_query(query_embedding, candidate_embeddings, top_k)
    
    def enhance_reasoning_embeddings(self, reasoning_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Enhance reasoning embeddings through denoising"""
        if not self.denoising_layer:
            logger.warning("Denoising layer not enabled")
            return torch.stack(reasoning_embeddings).mean(dim=0) if reasoning_embeddings else torch.zeros(768)
            
        return self.denoising_layer.enhance_reasoning_embeddings(reasoning_embeddings)
    
    # Stage 3: Planning and Reflection
    def generate_diverse_research_plan(self, research_question: str, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate diverse research plan using diffusion sampling"""
        if not self.planning_diffusion:
            logger.warning("Planning diffusion not enabled")
            # Fallback to simple plan
            return {
                'research_question': research_question,
                'selected_plan': {
                    'plan_steps': [
                        {'step_id': 1, 'description': f'Research {research_question}', 'tools_suggested': ['web_search']}
                    ],
                    'generation_method': 'fallback'
                },
                'planning_confidence': 0.5
            }
            
        return self.planning_diffusion.generate_research_plan(research_question, context)
    
    def refine_failed_trajectory(self, trajectory: List[Dict[str, Any]], 
                               failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Refine failed execution trajectory using diffusion"""
        if not self.planning_diffusion:
            logger.warning("Planning diffusion not enabled")
            return {
                'original_trajectory': trajectory,
                'refined_trajectory': trajectory,
                'refinement_applied': False
            }
            
        return self.planning_diffusion.refine_execution_trajectory(trajectory, failure_info)
    
    # Stage 4: Vision and Creativity
    def analyze_webpage_visually(self, url: str, analysis_focus: str = "content") -> Dict[str, Any]:
        """Analyze webpage with visual processing and enhancement"""
        if not self.image_driven_web_agent:
            logger.warning("Image-driven web agent not enabled")
            return {'error': 'Visual analysis not available'}
            
        return self.image_driven_web_agent.analyze_webpage_visually(url, analysis_focus)
    
    def explore_creative_ideas(self, initial_prompt: str, 
                             exploration_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explore and expand research ideas using diffusion-based creativity"""
        if not self.idea_exploration_agent:
            logger.warning("Idea exploration agent not enabled")
            return {
                'initial_prompt': initial_prompt,
                'idea_variations': [initial_prompt],
                'expanded_ideas': [],
                'novel_angles': [],
                'ranked_ideas': []
            }
            
        return self.idea_exploration_agent.explore_research_ideas(initial_prompt, exploration_context)
    
    # Stage 5: Enhanced RLHF and Alignment
    def generate_aligned_content(self, prompt: str, 
                               style_preference: str = "neutral",
                               quality_preference: str = "high") -> Dict[str, Any]:
        """Generate content aligned with human preferences"""
        if not self.rlhf_integrator:
            logger.warning("RLHF integration not enabled")
            return {
                'prompt': prompt,
                'generated_text': f"Response to: {prompt}",
                'alignment_score': 0.5,
                'generation_method': 'fallback'
            }
            
        return self.rlhf_integrator.generate_aligned_content(prompt, style_preference, quality_preference)
    
    def create_adversarial_training_examples(self, clean_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create adversarial examples for robustness training"""
        if not self.rlhf_integrator:
            logger.warning("RLHF integration not enabled")
            return []
            
        return self.rlhf_integrator.adversarial_creator.create_adversarial_examples(
            clean_examples, self.rlhf_integrator.classifier_free_guidance.style_classifier
        )
    
    # Integration utilities
    def get_diffusion_capabilities(self) -> Dict[str, bool]:
        """Get available diffusion capabilities"""
        return {
            'synthetic_data_generation': self.synthetic_data_generator is not None,
            'denoising_layer': self.denoising_layer is not None,
            'planning_diffusion': self.planning_diffusion is not None,
            'vision_creativity': self.image_driven_web_agent is not None and self.idea_exploration_agent is not None,
            'rlhf_integration': self.rlhf_integrator is not None,
            'core_diffusion_trained': self.diffusion_core.is_trained,
            'agent_initialized': self.is_initialized
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all diffusion components"""
        stats = {
            'core_diffusion': {
                'model_trained': self.diffusion_core.is_trained,
                'model_dim': self.diffusion_core.config.model_dim,
                'num_timesteps': self.diffusion_core.config.num_timesteps
            },
            'capabilities': self.get_diffusion_capabilities(),
            'training_contexts_count': len(self.training_contexts)
        }
        
        # Add component-specific stats
        if self.synthetic_data_generator:
            stats['synthetic_data'] = self.synthetic_data_generator.get_generation_stats()
            
        if self.denoising_layer:
            stats['denoising'] = self.denoising_layer.get_denoising_stats()
            
        if self.rlhf_integrator:
            stats['rlhf_integration'] = self.rlhf_integrator.get_integration_stats()
            
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.image_driven_web_agent:
            self.image_driven_web_agent.cleanup()
            
        logger.info("Diffusion-enhanced research agent cleanup completed")

# Factory function for easy integration
def create_diffusion_enhanced_agent(enable_all: bool = True, 
                                  custom_config: DiffusionIntegrationConfig = None) -> DiffusionEnhancedResearchAgent:
    """Factory function to create a diffusion-enhanced research agent"""
    if custom_config:
        config = custom_config
    elif enable_all:
        config = DiffusionIntegrationConfig(
            enable_synthetic_data=True,
            enable_denoising=True,
            enable_planning_diffusion=True,
            enable_vision_creativity=True,
            enable_rlhf_integration=True
        )
    else:
        config = DiffusionIntegrationConfig()
        
    return DiffusionEnhancedResearchAgent(config)