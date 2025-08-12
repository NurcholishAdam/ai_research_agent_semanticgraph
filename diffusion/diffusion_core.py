# -*- coding: utf-8 -*-
"""
Core Diffusion Framework for AI Research Agent
Provides foundational diffusion model capabilities for text and multimodal generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for diffusion models"""
    model_dim: int = 768
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # linear, cosine, sigmoid
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_sequence_length: int = 512

class NoiseScheduler:
    """Handles noise scheduling for diffusion process"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule based on configuration"""
        if self.config.noise_schedule == "linear":
            return torch.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                self.config.num_timesteps
            )
        elif self.config.noise_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.config.noise_schedule == "sigmoid":
            return self._sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine noise schedule for better quality"""
        timesteps = self.config.num_timesteps
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Sigmoid noise schedule"""
        timesteps = self.config.num_timesteps
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (self.config.beta_end - self.config.beta_start) + self.config.beta_start

class TextDiffusionModel(nn.Module):
    """Text diffusion model for embedding space"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim)
        )
        
        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim * 2),
            nn.LayerNorm(config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.model_dim * 2),
            nn.LayerNorm(config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.model_dim)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of diffusion model"""
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.config.model_dim)
        t_emb = self.time_embed(t_emb)
        
        # Combine input with time embedding
        h = torch.cat([x, t_emb], dim=-1)
        
        # Add conditioning if provided
        if condition is not None:
            h = torch.cat([h, condition], dim=-1)
            
        return self.denoiser(h)
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class DiffusionCore:
    """Core diffusion functionality for the research agent"""
    
    def __init__(self, config: DiffusionConfig = None):
        self.config = config or DiffusionConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.noise_scheduler = NoiseScheduler(self.config)
        self.model = TextDiffusionModel(self.config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
        self.embedding_model = AutoModel.from_pretrained(self.config.embedding_model).to(self.device)
        
        # Training state
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.is_trained = False
        
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to embeddings at timestep t"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.noise_scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.noise_scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x, noise
    
    def denoise_step(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single denoising step"""
        with torch.no_grad():
            predicted_noise = self.model(x, t, condition)
            
            alpha_t = self.noise_scheduler.alphas[t].view(-1, 1)
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1)
            beta_t = self.noise_scheduler.betas[t].view(-1, 1)
            
            # Compute denoised sample
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if t[0] > 0:
                alpha_cumprod_prev = self.noise_scheduler.alphas_cumprod[t - 1].view(-1, 1)
                pred_x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                return pred_x_prev
            else:
                return pred_x0
    
    def sample(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor] = None, num_steps: Optional[int] = None) -> torch.Tensor:
        """Sample from the diffusion model"""
        if num_steps is None:
            num_steps = self.config.num_timesteps
            
        # Start with pure noise
        x = torch.randn(shape, device=self.device)
        
        # Reverse diffusion process
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=self.device)
        
        for t in timesteps:
            t_batch = t.repeat(shape[0])
            x = self.denoise_step(x, t_batch, condition)
            
        return x
    
    def train_step(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> float:
        """Single training step"""
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noisy_x, noise = self.add_noise(x, t)
        
        # Predict noise
        predicted_noise = self.model(noisy_x, t, condition)
        
        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, training_data: List[str], num_epochs: int = 100, batch_size: int = 8):
        """Train the diffusion model"""
        logger.info(f"Training diffusion model for {num_epochs} epochs")
        
        # Encode training data
        embeddings = self.encode_text(training_data)
        dataset = torch.utils.data.TensorDataset(embeddings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                loss = self.train_step(x)
                total_loss += loss
                
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("Diffusion model training completed")
    
    def generate_variations(self, seed_texts: List[str], num_variations: int = 5, guidance_scale: float = 1.0) -> List[torch.Tensor]:
        """Generate variations of seed texts"""
        if not self.is_trained:
            logger.warning("Model not trained, using random sampling")
            
        # Encode seed texts as conditions
        conditions = self.encode_text(seed_texts)
        
        variations = []
        for _ in range(num_variations):
            # Sample with conditioning
            shape = (len(seed_texts), self.config.model_dim)
            variation = self.sample(shape, conditions)
            variations.append(variation)
            
        return variations