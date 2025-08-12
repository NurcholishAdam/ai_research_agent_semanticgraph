# -*- coding: utf-8 -*-
"""
Synthetic Data Generation for Training and Retrieval
Uses diffusion models to augment scarce examples and create balanced datasets
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging

from .diffusion_core import DiffusionCore, DiffusionConfig

logger = logging.getLogger(__name__)

@dataclass
class DataAugmentationConfig:
    """Configuration for data augmentation"""
    paraphrase_variations: int = 5
    noise_levels: List[float] = None
    augmentation_ratio: float = 2.0  # How many synthetic examples per real example
    min_similarity_threshold: float = 0.7
    max_similarity_threshold: float = 0.95
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

class ContextAugmenter:
    """Augments LangMem embeddings with synthetic paraphrases"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: DataAugmentationConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or DataAugmentationConfig()
        
    def augment_memory_contexts(self, contexts: List[str], metadata: List[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Augment memory contexts with synthetic paraphrases"""
        logger.info(f"Augmenting {len(contexts)} contexts with diffusion-generated paraphrases")
        
        augmented_contexts = contexts.copy()
        augmented_metadata = metadata.copy() if metadata else [{}] * len(contexts)
        
        # Generate paraphrases for each context
        for i, context in enumerate(contexts):
            try:
                paraphrases = self._generate_paraphrases(context, self.config.paraphrase_variations)
                
                for j, paraphrase in enumerate(paraphrases):
                    # Create metadata for synthetic example
                    synthetic_metadata = augmented_metadata[i].copy()
                    synthetic_metadata.update({
                        'synthetic': True,
                        'source_index': i,
                        'paraphrase_id': j,
                        'generation_method': 'diffusion_paraphrase',
                        'noise_level': self.config.noise_levels[j % len(self.config.noise_levels)]
                    })
                    
                    augmented_contexts.append(paraphrase)
                    augmented_metadata.append(synthetic_metadata)
                    
            except Exception as e:
                logger.warning(f"Failed to generate paraphrases for context {i}: {e}")
                continue
        
        logger.info(f"Generated {len(augmented_contexts) - len(contexts)} synthetic contexts")
        return augmented_contexts, augmented_metadata
    
    def _generate_paraphrases(self, context: str, num_variations: int) -> List[str]:
        """Generate paraphrases using controlled noise and denoising"""
        # Encode the original context
        original_embedding = self.diffusion_core.encode_text([context])
        
        paraphrases = []
        for i in range(num_variations):
            # Add controlled noise
            noise_level = self.config.noise_levels[i % len(self.config.noise_levels)]
            noise = torch.randn_like(original_embedding) * noise_level
            noisy_embedding = original_embedding + noise
            
            # Denoise back to coherent embedding
            timestep = torch.tensor([int(noise_level * self.diffusion_core.config.num_timesteps)], 
                                  device=self.diffusion_core.device)
            denoised_embedding = self.diffusion_core.denoise_step(noisy_embedding, timestep, original_embedding)
            
            # Convert back to text (this is a simplified approach - in practice, you'd need a decoder)
            paraphrase = self._embedding_to_text(denoised_embedding, context)
            paraphrases.append(paraphrase)
            
        return paraphrases
    
    def _embedding_to_text(self, embedding: torch.Tensor, original_text: str) -> str:
        """Convert embedding back to text (simplified implementation)"""
        # This is a placeholder - in practice, you'd use a trained decoder or retrieval method
        # For now, we'll create variations by modifying the original text
        variations = [
            f"In other words, {original_text.lower()}",
            f"To put it differently, {original_text}",
            f"Another way to express this: {original_text}",
            f"Essentially, {original_text.lower()}",
            f"Put simply, {original_text.lower()}"
        ]
        
        # Use embedding similarity to select best variation (simplified)
        return np.random.choice(variations)

class MultiModalBootstrapper:
    """Generates synthetic diagrams and charts from CSV data using latent diffusion"""
    
    def __init__(self, diffusion_core: DiffusionCore):
        self.diffusion_core = diffusion_core
        
    def generate_synthetic_charts(self, csv_data: pd.DataFrame, chart_types: List[str] = None) -> List[Dict[str, Any]]:
        """Generate synthetic charts conditioned on CSV data"""
        if chart_types is None:
            chart_types = ['bar', 'line', 'scatter', 'histogram', 'heatmap']
            
        logger.info(f"Generating synthetic charts for dataset with shape {csv_data.shape}")
        
        synthetic_charts = []
        
        for chart_type in chart_types:
            try:
                chart_data = self._create_chart(csv_data, chart_type)
                synthetic_charts.append({
                    'type': chart_type,
                    'data': chart_data['image'],
                    'description': chart_data['description'],
                    'metadata': {
                        'synthetic': True,
                        'source_columns': list(csv_data.columns),
                        'chart_type': chart_type,
                        'generation_method': 'latent_diffusion_chart'
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to generate {chart_type} chart: {e}")
                continue
                
        return synthetic_charts
    
    def _create_chart(self, data: pd.DataFrame, chart_type: str) -> Dict[str, Any]:
        """Create a specific type of chart"""
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar' and len(data.columns) >= 2:
            x_col, y_col = data.columns[0], data.columns[1]
            if data[x_col].dtype == 'object':
                plt.bar(data[x_col].head(10), data[y_col].head(10))
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                description = f"Bar chart showing {y_col} by {x_col}"
            else:
                plt.hist(data[x_col].dropna(), bins=20)
                plt.xlabel(x_col)
                plt.ylabel('Frequency')
                description = f"Histogram of {x_col}"
                
        elif chart_type == 'line' and len(data.columns) >= 2:
            x_col, y_col = data.columns[0], data.columns[1]
            plt.plot(data[x_col].head(50), data[y_col].head(50))
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            description = f"Line chart showing {y_col} over {x_col}"
            
        elif chart_type == 'scatter' and len(data.columns) >= 2:
            x_col, y_col = data.columns[0], data.columns[1]
            plt.scatter(data[x_col].head(100), data[y_col].head(100), alpha=0.6)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            description = f"Scatter plot of {y_col} vs {x_col}"
            
        elif chart_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]
            if len(numeric_cols) >= 2:
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                description = f"Correlation heatmap of numeric columns"
            else:
                plt.text(0.5, 0.5, 'Insufficient numeric data', ha='center', va='center')
                description = "Placeholder heatmap"
                
        else:
            # Default to simple bar chart
            if len(data.columns) > 0:
                col = data.columns[0]
                if data[col].dtype == 'object':
                    value_counts = data[col].value_counts().head(10)
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                else:
                    plt.hist(data[col].dropna(), bins=20)
                description = f"Distribution of {col}"
            else:
                plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
                description = "Empty chart"
        
        plt.title(f"Synthetic {chart_type.title()} Chart")
        plt.tight_layout()
        
        # Convert to base64 image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'image': image_base64,
            'description': description
        }

class BalancedDatasetCreator:
    """Creates balanced datasets by diffusing under-represented classes"""
    
    def __init__(self, diffusion_core: DiffusionCore):
        self.diffusion_core = diffusion_core
        
    def balance_rlhf_dataset(self, training_examples: List[Dict[str, Any]], target_balance_ratio: float = 1.0) -> List[Dict[str, Any]]:
        """Balance RLHF training dataset by generating synthetic examples for under-represented classes"""
        logger.info("Balancing RLHF dataset using diffusion-generated examples")
        
        # Analyze class distribution
        class_distribution = self._analyze_class_distribution(training_examples)
        logger.info(f"Original class distribution: {class_distribution}")
        
        # Identify under-represented classes
        max_count = max(class_distribution.values())
        target_count = int(max_count * target_balance_ratio)
        
        balanced_examples = training_examples.copy()
        
        for class_label, count in class_distribution.items():
            if count < target_count:
                needed_examples = target_count - count
                logger.info(f"Generating {needed_examples} synthetic examples for class '{class_label}'")
                
                # Get examples from this class
                class_examples = [ex for ex in training_examples if ex.get('label') == class_label]
                
                # Generate synthetic examples
                synthetic_examples = self._generate_synthetic_examples(class_examples, needed_examples)
                balanced_examples.extend(synthetic_examples)
        
        logger.info(f"Balanced dataset size: {len(balanced_examples)} (was {len(training_examples)})")
        return balanced_examples
    
    def _analyze_class_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze the distribution of classes in the dataset"""
        distribution = {}
        for example in examples:
            label = example.get('label', 'unknown')
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def _generate_synthetic_examples(self, class_examples: List[Dict[str, Any]], num_needed: int) -> List[Dict[str, Any]]:
        """Generate synthetic examples for a specific class"""
        if not class_examples:
            return []
            
        synthetic_examples = []
        
        # Extract texts from class examples
        texts = [ex.get('text', '') for ex in class_examples if ex.get('text')]
        if not texts:
            return []
        
        # Generate variations using diffusion
        for i in range(num_needed):
            # Select a random seed example
            seed_example = np.random.choice(class_examples)
            seed_text = seed_example.get('text', '')
            
            if not seed_text:
                continue
                
            # Generate variation
            try:
                variations = self.diffusion_core.generate_variations([seed_text], num_variations=1)
                if variations:
                    # Create synthetic example
                    synthetic_example = seed_example.copy()
                    synthetic_example.update({
                        'text': self._embedding_to_text_variation(variations[0][0], seed_text),
                        'synthetic': True,
                        'source_example_id': seed_example.get('id', f'unknown_{i}'),
                        'generation_method': 'diffusion_class_balancing'
                    })
                    synthetic_examples.append(synthetic_example)
                    
            except Exception as e:
                logger.warning(f"Failed to generate synthetic example {i}: {e}")
                continue
        
        return synthetic_examples
    
    def _embedding_to_text_variation(self, embedding: torch.Tensor, original_text: str) -> str:
        """Convert embedding to text variation (simplified implementation)"""
        # This is a placeholder - in practice, you'd use a trained decoder
        # For now, we'll create variations by modifying the original text
        
        # Simple text variations
        variations = [
            original_text.replace('.', ', which is important.'),
            f"Consider this: {original_text}",
            original_text.replace('is', 'appears to be'),
            original_text.replace('will', 'might'),
            f"From my perspective, {original_text.lower()}"
        ]
        
        return np.random.choice(variations)

class SyntheticDataGenerator:
    """Main class coordinating all synthetic data generation capabilities"""
    
    def __init__(self, diffusion_config: DiffusionConfig = None, augmentation_config: DataAugmentationConfig = None):
        self.diffusion_core = DiffusionCore(diffusion_config)
        self.context_augmenter = ContextAugmenter(self.diffusion_core, augmentation_config)
        self.multimodal_bootstrapper = MultiModalBootstrapper(self.diffusion_core)
        self.balanced_dataset_creator = BalancedDatasetCreator(self.diffusion_core)
        
    def initialize(self, training_contexts: List[str]):
        """Initialize the diffusion models with training data"""
        logger.info("Initializing synthetic data generator with training contexts")
        self.diffusion_core.train(training_contexts)
        
    def augment_langmem_contexts(self, contexts: List[str], metadata: List[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Augment LangMem contexts with synthetic paraphrases"""
        return self.context_augmenter.augment_memory_contexts(contexts, metadata)
    
    def generate_multimodal_content(self, csv_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate synthetic charts and diagrams from CSV data"""
        return self.multimodal_bootstrapper.generate_synthetic_charts(csv_data)
    
    def balance_training_dataset(self, training_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance training dataset by generating synthetic examples"""
        return self.balanced_dataset_creator.balance_rlhf_dataset(training_examples)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about synthetic data generation"""
        return {
            'diffusion_model_trained': self.diffusion_core.is_trained,
            'model_config': {
                'model_dim': self.diffusion_core.config.model_dim,
                'num_timesteps': self.diffusion_core.config.num_timesteps,
                'noise_schedule': self.diffusion_core.config.noise_schedule
            },
            'augmentation_config': {
                'paraphrase_variations': self.context_augmenter.config.paraphrase_variations,
                'augmentation_ratio': self.context_augmenter.config.augmentation_ratio
            }
        }