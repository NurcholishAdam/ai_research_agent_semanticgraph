#!/usr/bin/env python3
"""
Diffusion Models Integration for AI Research Agent
Provides synthetic data generation, denoising, planning, and creative capabilities
"""

from .diffusion_core import DiffusionCore, DiffusionConfig
from .synthetic_data_generator import SyntheticDataGenerator, DataAugmentationConfig
from .denoising_layer import DenoisingLayer, NoiseConfig
from .planning_diffusion import PlanningDiffusion, PlanningConfig
from .vision_creativity_agents import ImageDrivenWebAgent, IdeaExplorationAgent
from .rlhf_diffusion_integration import RLHFDiffusionIntegrator, AlignmentConfig

__all__ = [
    'DiffusionCore',
    'DiffusionConfig',
    'SyntheticDataGenerator',
    'DataAugmentationConfig',
    'DenoisingLayer',
    'NoiseConfig',
    'PlanningDiffusion',
    'PlanningConfig',
    'ImageDrivenWebAgent',
    'IdeaExplorationAgent',
    'RLHFDiffusionIntegrator',
    'AlignmentConfig'
]

__version__ = "1.0.0"
__author__ = "AI Research Agent Team"
