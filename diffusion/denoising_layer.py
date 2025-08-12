# -*- coding: utf-8 -*-
"""
Denoising Layer for Retrieval and Reasoning
Provides noise robustness and retrieval-augmented diffusion capabilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

from .diffusion_core import DiffusionCore, DiffusionConfig

logger = logging.getLogger(__name__)

@dataclass
class NoiseConfig:
    """Configuration for noise injection and denoising"""
    gaussian_noise_std: float = 0.1
    noise_injection_probability: float = 0.3
    denoising_steps: int = 5
    robustness_test_noise_levels: List[float] = None
    retrieval_denoising_strength: float = 0.2
    coherence_threshold: float = 0.8
    
    def __post_init__(self):
        if self.robustness_test_noise_levels is None:
            self.robustness_test_noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]

class NoisyPromptRobustness:
    """Handles noisy prompt robustness through controlled noise injection and denoising"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: NoiseConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or NoiseConfig()
        
    def add_controlled_noise(self, embeddings: torch.Tensor, noise_level: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add controlled Gaussian noise to embeddings"""
        if noise_level is None:
            noise_level = self.config.gaussian_noise_std
            
        noise = torch.randn_like(embeddings) * noise_level
        noisy_embeddings = embeddings + noise
        
        return noisy_embeddings, noise
    
    def denoise_embeddings(self, noisy_embeddings: torch.Tensor, num_steps: int = None) -> torch.Tensor:
        """Apply reverse diffusion steps to denoise embeddings"""
        if num_steps is None:
            num_steps = self.config.denoising_steps
            
        logger.debug(f"Denoising embeddings with {num_steps} reverse diffusion steps")
        
        current_embeddings = noisy_embeddings
        
        # Apply reverse diffusion steps
        for step in range(num_steps):
            # Calculate timestep (from high noise to low noise)
            t = torch.tensor([int((num_steps - step - 1) * self.diffusion_core.config.num_timesteps / num_steps)], 
                           device=self.diffusion_core.device)
            t_batch = t.repeat(current_embeddings.shape[0])
            
            # Apply denoising step
            current_embeddings = self.diffusion_core.denoise_step(current_embeddings, t_batch)
            
        return current_embeddings
    
    def test_robustness(self, clean_embeddings: torch.Tensor, test_function: callable) -> Dict[str, float]:
        """Test robustness against different noise levels"""
        logger.info("Testing noise robustness across different noise levels")
        
        robustness_results = {}
        
        for noise_level in self.config.robustness_test_noise_levels:
            # Add noise
            noisy_embeddings, _ = self.add_controlled_noise(clean_embeddings, noise_level)
            
            # Denoise
            denoised_embeddings = self.denoise_embeddings(noisy_embeddings)
            
            # Test performance
            try:
                performance = test_function(denoised_embeddings)
                robustness_results[f'noise_{noise_level}'] = performance
                logger.debug(f"Performance at noise level {noise_level}: {performance}")
            except Exception as e:
                logger.warning(f"Failed to test at noise level {noise_level}: {e}")
                robustness_results[f'noise_{noise_level}'] = 0.0
                
        return robustness_results
    
    def robust_embedding_retrieval(self, query_embedding: torch.Tensor, candidate_embeddings: torch.Tensor, 
                                 top_k: int = 5) -> Tuple[torch.Tensor, List[int]]:
        """Perform robust retrieval with noise injection and denoising"""
        # Add noise to query
        if np.random.random() < self.config.noise_injection_probability:
            noisy_query, _ = self.add_controlled_noise(query_embedding)
            denoised_query = self.denoise_embeddings(noisy_query)
        else:
            denoised_query = query_embedding
            
        # Compute similarities
        similarities = torch.cosine_similarity(
            denoised_query.unsqueeze(0), 
            candidate_embeddings, 
            dim=1
        )
        
        # Get top-k results
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        return top_k_values, top_k_indices.tolist()

class RetrievalAugmentedDiffusion:
    """Implements retrieval-augmented diffusion for coherent summary embeddings"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: NoiseConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or NoiseConfig()
        
    def create_coherent_summary(self, mixed_embeddings: List[torch.Tensor], 
                              weights: Optional[List[float]] = None) -> torch.Tensor:
        """Create coherent summary embedding from mixed embeddings using diffusion"""
        logger.debug(f"Creating coherent summary from {len(mixed_embeddings)} mixed embeddings")
        
        if not mixed_embeddings:
            raise ValueError("No embeddings provided for summary creation")
            
        # Stack embeddings
        stacked_embeddings = torch.stack(mixed_embeddings)
        
        # Apply weights if provided
        if weights:
            weights_tensor = torch.tensor(weights, device=stacked_embeddings.device).unsqueeze(-1)
            weighted_embeddings = stacked_embeddings * weights_tensor
        else:
            weighted_embeddings = stacked_embeddings
            
        # Create noisy mixture
        mixture = torch.mean(weighted_embeddings, dim=0, keepdim=True)
        
        # Add noise to simulate mixture uncertainty
        noisy_mixture, _ = self.add_mixture_noise(mixture)
        
        # Denoise to create coherent summary
        coherent_summary = self.denoise_mixture(noisy_mixture, mixed_embeddings)
        
        return coherent_summary.squeeze(0)
    
    def add_mixture_noise(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to embedding mixture"""
        noise_level = self.config.retrieval_denoising_strength
        noise = torch.randn_like(mixture) * noise_level
        noisy_mixture = mixture + noise
        
        return noisy_mixture, noise
    
    def denoise_mixture(self, noisy_mixture: torch.Tensor, 
                       context_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Denoise mixture using context embeddings as guidance"""
        # Stack context embeddings for conditioning
        if context_embeddings:
            context_stack = torch.stack(context_embeddings)
            context_condition = torch.mean(context_stack, dim=0, keepdim=True)
        else:
            context_condition = None
            
        # Apply denoising steps with context conditioning
        current_mixture = noisy_mixture
        
        for step in range(self.config.denoising_steps):
            # Calculate timestep
            t = torch.tensor([int((self.config.denoising_steps - step - 1) * 
                                self.diffusion_core.config.num_timesteps / self.config.denoising_steps)], 
                           device=self.diffusion_core.device)
            t_batch = t.repeat(current_mixture.shape[0])
            
            # Apply denoising with context conditioning
            current_mixture = self.diffusion_core.denoise_step(current_mixture, t_batch, context_condition)
            
        return current_mixture
    
    def assess_coherence(self, summary_embedding: torch.Tensor, 
                        source_embeddings: List[torch.Tensor]) -> float:
        """Assess coherence of summary embedding with source embeddings"""
        if not source_embeddings:
            return 0.0
            
        # Calculate similarities with all source embeddings
        similarities = []
        for source_emb in source_embeddings:
            sim = torch.cosine_similarity(summary_embedding.unsqueeze(0), source_emb.unsqueeze(0))
            similarities.append(sim.item())
            
        # Return average similarity as coherence score
        coherence_score = np.mean(similarities)
        return coherence_score
    
    def iterative_refinement(self, initial_summary: torch.Tensor, 
                           source_embeddings: List[torch.Tensor],
                           max_iterations: int = 3) -> torch.Tensor:
        """Iteratively refine summary embedding for better coherence"""
        current_summary = initial_summary
        best_summary = initial_summary
        best_coherence = self.assess_coherence(initial_summary, source_embeddings)
        
        logger.debug(f"Initial coherence: {best_coherence:.3f}")
        
        for iteration in range(max_iterations):
            # Add small amount of noise and denoise
            noisy_summary, _ = self.add_mixture_noise(current_summary.unsqueeze(0))
            refined_summary = self.denoise_mixture(noisy_summary, source_embeddings)
            
            # Assess coherence
            coherence = self.assess_coherence(refined_summary.squeeze(0), source_embeddings)
            
            logger.debug(f"Iteration {iteration + 1} coherence: {coherence:.3f}")
            
            # Keep best version
            if coherence > best_coherence:
                best_summary = refined_summary.squeeze(0)
                best_coherence = coherence
                
            current_summary = refined_summary.squeeze(0)
            
            # Early stopping if coherence is high enough
            if best_coherence >= self.config.coherence_threshold:
                logger.debug(f"Early stopping at iteration {iteration + 1} with coherence {best_coherence:.3f}")
                break
                
        return best_summary

class DenoisingLayer:
    """Main denoising layer integrating all denoising capabilities"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: NoiseConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or NoiseConfig()
        
        # Initialize components
        self.noisy_prompt_robustness = NoisyPromptRobustness(diffusion_core, config)
        self.retrieval_augmented_diffusion = RetrievalAugmentedDiffusion(diffusion_core, config)
        
    def process_retrieval_query(self, query_embedding: torch.Tensor, 
                              candidate_embeddings: torch.Tensor,
                              top_k: int = 5) -> Dict[str, Any]:
        """Process retrieval query with noise robustness"""
        logger.debug("Processing retrieval query with denoising layer")
        
        # Perform robust retrieval
        similarities, indices = self.noisy_prompt_robustness.robust_embedding_retrieval(
            query_embedding, candidate_embeddings, top_k
        )
        
        # Get retrieved embeddings
        retrieved_embeddings = [candidate_embeddings[i] for i in indices]
        
        # Create coherent summary
        summary_embedding = self.retrieval_augmented_diffusion.create_coherent_summary(
            [query_embedding] + retrieved_embeddings
        )
        
        # Assess coherence
        coherence_score = self.retrieval_augmented_diffusion.assess_coherence(
            summary_embedding, retrieved_embeddings
        )
        
        return {
            'retrieved_indices': indices,
            'similarities': similarities.tolist(),
            'summary_embedding': summary_embedding,
            'coherence_score': coherence_score,
            'denoised': True
        }
    
    def enhance_reasoning_embeddings(self, reasoning_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Enhance reasoning embeddings through denoising and coherence optimization"""
        logger.debug(f"Enhancing {len(reasoning_embeddings)} reasoning embeddings")
        
        if not reasoning_embeddings:
            raise ValueError("No reasoning embeddings provided")
            
        # Create initial summary
        initial_summary = self.retrieval_augmented_diffusion.create_coherent_summary(reasoning_embeddings)
        
        # Iteratively refine for better coherence
        enhanced_summary = self.retrieval_augmented_diffusion.iterative_refinement(
            initial_summary, reasoning_embeddings
        )
        
        return enhanced_summary
    
    def test_system_robustness(self, test_embeddings: torch.Tensor, 
                             evaluation_function: callable) -> Dict[str, Any]:
        """Test overall system robustness to noise"""
        logger.info("Testing denoising layer robustness")
        
        # Test noisy prompt robustness
        robustness_results = self.noisy_prompt_robustness.test_robustness(
            test_embeddings, evaluation_function
        )
        
        # Test coherence maintenance
        coherence_scores = []
        for i in range(min(10, len(test_embeddings))):  # Test on subset
            embedding = test_embeddings[i:i+1]
            noisy_embedding, _ = self.noisy_prompt_robustness.add_controlled_noise(embedding)
            denoised_embedding = self.noisy_prompt_robustness.denoise_embeddings(noisy_embedding)
            
            coherence = torch.cosine_similarity(embedding, denoised_embedding).item()
            coherence_scores.append(coherence)
            
        return {
            'noise_robustness': robustness_results,
            'average_coherence_preservation': np.mean(coherence_scores),
            'coherence_std': np.std(coherence_scores),
            'config': {
                'gaussian_noise_std': self.config.gaussian_noise_std,
                'denoising_steps': self.config.denoising_steps,
                'coherence_threshold': self.config.coherence_threshold
            }
        }
    
    def get_denoising_stats(self) -> Dict[str, Any]:
        """Get statistics about denoising operations"""
        return {
            'config': {
                'gaussian_noise_std': self.config.gaussian_noise_std,
                'noise_injection_probability': self.config.noise_injection_probability,
                'denoising_steps': self.config.denoising_steps,
                'retrieval_denoising_strength': self.config.retrieval_denoising_strength,
                'coherence_threshold': self.config.coherence_threshold
            },
            'robustness_test_levels': self.config.robustness_test_noise_levels,
            'diffusion_model_ready': self.diffusion_core.is_trained
        }