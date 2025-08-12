# -*- coding: utf-8 -*-
"""
Vision and Creativity Sub-Agents
Implements image-driven web agent and idea exploration agent using diffusion models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import base64
import io
import logging
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2

from .diffusion_core import DiffusionCore, DiffusionConfig

logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for vision and creativity agents"""
    image_resolution: Tuple[int, int] = (1024, 768)
    annotation_font_size: int = 16
    highlight_colors: List[str] = None
    creativity_temperature: float = 0.8
    idea_expansion_depth: int = 3
    brainstorming_iterations: int = 5
    screenshot_quality: int = 95
    
    def __post_init__(self):
        if self.highlight_colors is None:
            self.highlight_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

class ImageDiffusionProcessor:
    """Processes images using diffusion models for enhancement and annotation"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: VisionConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or VisionConfig()
        
    def enhance_screenshot(self, screenshot_data: bytes, enhancement_type: str = "highlight") -> Dict[str, Any]:
        """Enhance screenshot with diffusion-based processing"""
        logger.debug(f"Enhancing screenshot with {enhancement_type} enhancement")
        
        # Load image
        image = Image.open(io.BytesIO(screenshot_data))
        
        # Resize to target resolution
        image = image.resize(self.config.image_resolution, Image.Resampling.LANCZOS)
        
        # Apply enhancement based on type
        if enhancement_type == "highlight":
            enhanced_image = self._apply_highlight_enhancement(image)
        elif enhancement_type == "annotate":
            enhanced_image = self._apply_annotation_enhancement(image)
        elif enhancement_type == "focus":
            enhanced_image = self._apply_focus_enhancement(image)
        else:
            enhanced_image = image
            
        # Convert back to bytes
        output_buffer = io.BytesIO()
        enhanced_image.save(output_buffer, format='PNG', quality=self.config.screenshot_quality)
        enhanced_data = output_buffer.getvalue()
        
        return {
            'enhanced_image': base64.b64encode(enhanced_data).decode(),
            'original_size': image.size,
            'enhanced_size': enhanced_image.size,
            'enhancement_type': enhancement_type,
            'processing_method': 'diffusion_enhancement'
        }
    
    def _apply_highlight_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply highlight enhancement to emphasize important regions"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Simple region detection (in practice, use more sophisticated methods)
        # For now, we'll highlight regions with high contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours for highlighting
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create enhanced image
        enhanced = image.copy()
        draw = ImageDraw.Draw(enhanced)
        
        # Highlight important regions
        for i, contour in enumerate(contours[:10]):  # Limit to top 10 regions
            if cv2.contourArea(contour) > 1000:  # Only large regions
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw highlight rectangle
                color = self.config.highlight_colors[i % len(self.config.highlight_colors)]
                draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
                
        return enhanced
    
    def _apply_annotation_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply annotation enhancement with text labels"""
        enhanced = image.copy()
        draw = ImageDraw.Draw(enhanced)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", self.config.annotation_font_size)
        except:
            font = ImageFont.load_default()
            
        # Add sample annotations (in practice, use content analysis)
        annotations = [
            {"text": "Main Content", "position": (50, 50), "color": "#FF6B6B"},
            {"text": "Navigation", "position": (50, 100), "color": "#4ECDC4"},
            {"text": "Key Information", "position": (50, 150), "color": "#45B7D1"}
        ]
        
        for annotation in annotations:
            draw.text(
                annotation["position"], 
                annotation["text"], 
                fill=annotation["color"], 
                font=font
            )
            
        return enhanced
    
    def _apply_focus_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply focus enhancement to emphasize central content"""
        # Create a radial gradient mask for focus effect
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        
        # Create mask
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # Draw radial gradient
        max_radius = min(width, height) // 2
        for radius in range(max_radius, 0, -5):
            alpha = int(255 * (1 - radius / max_radius))
            mask_draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=alpha)
            
        # Apply focus effect
        enhanced = image.copy()
        enhancer = ImageEnhance.Brightness(enhanced)
        darkened = enhancer.enhance(0.5)
        
        # Composite with mask
        enhanced.paste(darkened, mask=mask)
        
        return enhanced

class WebScreenshotAgent:
    """Agent for taking and processing web screenshots"""
    
    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self.driver = None
        
    def initialize_driver(self):
        """Initialize web driver for screenshots"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"--window-size={self.config.image_resolution[0]},{self.config.image_resolution[1]}")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                logger.info("Web driver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize web driver: {e}")
                self.driver = None
                
    def capture_webpage(self, url: str) -> Optional[bytes]:
        """Capture screenshot of webpage"""
        if self.driver is None:
            self.initialize_driver()
            
        if self.driver is None:
            logger.error("Web driver not available")
            return None
            
        try:
            logger.info(f"Capturing screenshot of {url}")
            self.driver.get(url)
            
            # Wait for page to load
            self.driver.implicitly_wait(3)
            
            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            return screenshot
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
            
    def cleanup(self):
        """Cleanup web driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

class ImageDrivenWebAgent:
    """Web agent with image diffusion capabilities for visual content analysis"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: VisionConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or VisionConfig()
        self.image_processor = ImageDiffusionProcessor(diffusion_core, config)
        self.screenshot_agent = WebScreenshotAgent(config)
        
    def analyze_webpage_visually(self, url: str, analysis_focus: str = "content") -> Dict[str, Any]:
        """Analyze webpage with visual processing and enhancement"""
        logger.info(f"Performing visual analysis of {url} with focus on {analysis_focus}")
        
        # Capture screenshot
        screenshot_data = self.screenshot_agent.capture_webpage(url)
        if not screenshot_data:
            return {'error': 'Failed to capture webpage screenshot'}
            
        # Enhance screenshot based on analysis focus
        enhancement_type = self._determine_enhancement_type(analysis_focus)
        enhanced_result = self.image_processor.enhance_screenshot(screenshot_data, enhancement_type)
        
        # Analyze visual content
        visual_analysis = self._analyze_visual_content(screenshot_data, analysis_focus)
        
        # Generate insights
        insights = self._generate_visual_insights(visual_analysis, analysis_focus)
        
        return {
            'url': url,
            'analysis_focus': analysis_focus,
            'enhanced_image': enhanced_result['enhanced_image'],
            'visual_analysis': visual_analysis,
            'insights': insights,
            'enhancement_details': enhanced_result,
            'processing_method': 'diffusion_visual_analysis'
        }
    
    def _determine_enhancement_type(self, analysis_focus: str) -> str:
        """Determine appropriate enhancement type based on analysis focus"""
        focus_mapping = {
            'content': 'highlight',
            'navigation': 'annotate',
            'headlines': 'focus',
            'layout': 'highlight',
            'text': 'focus'
        }
        return focus_mapping.get(analysis_focus, 'highlight')
    
    def _analyze_visual_content(self, screenshot_data: bytes, focus: str) -> Dict[str, Any]:
        """Analyze visual content of screenshot"""
        # Load image for analysis
        image = Image.open(io.BytesIO(screenshot_data))
        img_array = np.array(image)
        
        # Basic visual analysis
        analysis = {
            'image_dimensions': image.size,
            'dominant_colors': self._extract_dominant_colors(img_array),
            'text_regions': self._detect_text_regions(img_array),
            'layout_structure': self._analyze_layout_structure(img_array),
            'visual_complexity': self._calculate_visual_complexity(img_array)
        }
        
        # Focus-specific analysis
        if focus == 'headlines':
            analysis['headline_regions'] = self._detect_headline_regions(img_array)
        elif focus == 'navigation':
            analysis['navigation_elements'] = self._detect_navigation_elements(img_array)
        elif focus == 'content':
            analysis['content_blocks'] = self._detect_content_blocks(img_array)
            
        return analysis
    
    def _extract_dominant_colors(self, img_array: np.ndarray) -> List[str]:
        """Extract dominant colors from image"""
        # Reshape image to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Simple color clustering (in practice, use k-means)
        unique_colors = np.unique(pixels, axis=0)
        
        # Convert to hex colors (sample first 5)
        hex_colors = []
        for color in unique_colors[:5]:
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            hex_colors.append(hex_color)
            
        return hex_colors
    
    def _detect_text_regions(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get text regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (likely text regions)
            if w > 20 and h > 10 and w < img_array.shape[1] * 0.8:
                text_regions.append({
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'aspect_ratio': w / h
                })
                
        return text_regions[:10]  # Return top 10 regions
    
    def _analyze_layout_structure(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze layout structure of webpage"""
        height, width = img_array.shape[:2]
        
        # Divide into grid sections for analysis
        grid_rows, grid_cols = 4, 3
        section_height = height // grid_rows
        section_width = width // grid_cols
        
        layout_analysis = {
            'grid_structure': f"{grid_rows}x{grid_cols}",
            'sections': [],
            'content_distribution': {}
        }
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                y1 = row * section_height
                y2 = (row + 1) * section_height
                x1 = col * section_width
                x2 = (col + 1) * section_width
                
                section = img_array[y1:y2, x1:x2]
                
                # Analyze section content
                section_analysis = {
                    'position': f"row_{row}_col_{col}",
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'content_density': self._calculate_content_density(section),
                    'color_variance': np.var(section)
                }
                
                layout_analysis['sections'].append(section_analysis)
                
        return layout_analysis
    
    def _calculate_visual_complexity(self, img_array: np.ndarray) -> float:
        """Calculate visual complexity score"""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate color variance
        color_variance = np.var(img_array)
        
        # Combine metrics
        complexity = (edge_density * 0.6 + (color_variance / 10000) * 0.4)
        return min(1.0, complexity)
    
    def _detect_headline_regions(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential headline regions"""
        # Headlines are typically larger text at top of page
        height = img_array.shape[0]
        top_section = img_array[:height//3, :]  # Top third of page
        
        # Detect text regions in top section
        text_regions = self._detect_text_regions(top_section)
        
        # Filter for headline-like regions (larger, wider)
        headlines = []
        for region in text_regions:
            bbox = region['bbox']
            if bbox[2] > 100 and bbox[3] > 20:  # Width > 100, height > 20
                headlines.append({
                    'bbox': bbox,
                    'confidence': min(1.0, (bbox[2] * bbox[3]) / 10000),
                    'type': 'headline'
                })
                
        return headlines
    
    def _detect_navigation_elements(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect navigation elements"""
        # Navigation typically at top or left side
        height, width = img_array.shape[:2]
        
        # Check top navigation
        top_nav = img_array[:height//8, :]
        left_nav = img_array[:, :width//6]
        
        nav_elements = []
        
        # Analyze top navigation
        top_regions = self._detect_text_regions(top_nav)
        for region in top_regions:
            nav_elements.append({
                'bbox': region['bbox'],
                'type': 'top_navigation',
                'confidence': 0.8
            })
            
        # Analyze left navigation
        left_regions = self._detect_text_regions(left_nav)
        for region in left_regions:
            nav_elements.append({
                'bbox': region['bbox'],
                'type': 'left_navigation',
                'confidence': 0.7
            })
            
        return nav_elements
    
    def _detect_content_blocks(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect main content blocks"""
        # Content blocks are typically in center area
        height, width = img_array.shape[:2]
        
        # Focus on center area
        content_area = img_array[height//4:3*height//4, width//6:5*width//6]
        
        # Detect regions in content area
        content_regions = self._detect_text_regions(content_area)
        
        content_blocks = []
        for region in content_regions:
            if region['area'] > 5000:  # Large enough to be content block
                content_blocks.append({
                    'bbox': region['bbox'],
                    'type': 'content_block',
                    'estimated_word_count': region['area'] // 100,
                    'confidence': min(1.0, region['area'] / 20000)
                })
                
        return content_blocks
    
    def _calculate_content_density(self, section: np.ndarray) -> float:
        """Calculate content density in image section"""
        # Simple content density based on pixel variance
        return min(1.0, np.var(section) / 10000)
    
    def _generate_visual_insights(self, analysis: Dict[str, Any], focus: str) -> List[str]:
        """Generate insights from visual analysis"""
        insights = []
        
        # General insights
        complexity = analysis.get('visual_complexity', 0)
        if complexity > 0.7:
            insights.append("High visual complexity detected - page may be information-dense")
        elif complexity < 0.3:
            insights.append("Low visual complexity - clean, minimal design")
            
        # Text region insights
        text_regions = analysis.get('text_regions', [])
        if len(text_regions) > 15:
            insights.append("Text-heavy page with multiple content sections")
        elif len(text_regions) < 5:
            insights.append("Minimal text content - likely image or media focused")
            
        # Focus-specific insights
        if focus == 'headlines' and 'headline_regions' in analysis:
            headlines = analysis['headline_regions']
            insights.append(f"Detected {len(headlines)} potential headline regions")
            
        elif focus == 'navigation' and 'navigation_elements' in analysis:
            nav_elements = analysis['navigation_elements']
            insights.append(f"Found {len(nav_elements)} navigation elements")
            
        elif focus == 'content' and 'content_blocks' in analysis:
            content_blocks = analysis['content_blocks']
            total_words = sum(block.get('estimated_word_count', 0) for block in content_blocks)
            insights.append(f"Estimated {total_words} words across {len(content_blocks)} content blocks")
            
        return insights
    
    def cleanup(self):
        """Cleanup resources"""
        self.screenshot_agent.cleanup()

class IdeaExplorationAgent:
    """Creative brainstorming agent using conditional text diffusion"""
    
    def __init__(self, diffusion_core: DiffusionCore, config: VisionConfig = None):
        self.diffusion_core = diffusion_core
        self.config = config or VisionConfig()
        
    def explore_research_ideas(self, initial_prompt: str, 
                             exploration_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explore and expand research ideas using diffusion-based creativity"""
        logger.info(f"Exploring research ideas from prompt: {initial_prompt}")
        
        # Generate diverse idea variations
        idea_variations = self._generate_idea_variations(initial_prompt, exploration_context)
        
        # Expand each variation
        expanded_ideas = []
        for variation in idea_variations:
            expansion = self._expand_idea_recursively(variation, self.config.idea_expansion_depth)
            expanded_ideas.append(expansion)
            
        # Synthesize novel angles
        novel_angles = self._synthesize_novel_angles(expanded_ideas, initial_prompt)
        
        # Rank ideas by creativity and feasibility
        ranked_ideas = self._rank_ideas_by_creativity(expanded_ideas + novel_angles)
        
        return {
            'initial_prompt': initial_prompt,
            'idea_variations': idea_variations,
            'expanded_ideas': expanded_ideas,
            'novel_angles': novel_angles,
            'ranked_ideas': ranked_ideas,
            'exploration_stats': {
                'total_ideas_generated': len(expanded_ideas) + len(novel_angles),
                'expansion_depth': self.config.idea_expansion_depth,
                'creativity_temperature': self.config.creativity_temperature
            }
        }
    
    def _generate_idea_variations(self, prompt: str, context: Dict[str, Any] = None) -> List[str]:
        """Generate variations of the initial idea using diffusion sampling"""
        # Encode the prompt
        prompt_embedding = self.diffusion_core.encode_text([prompt])
        
        # Generate variations with different noise levels
        variations = []
        
        for i in range(self.config.brainstorming_iterations):
            # Add creative noise
            creativity_noise = torch.randn_like(prompt_embedding) * self.config.creativity_temperature
            creative_embedding = prompt_embedding + creativity_noise
            
            # Denoise to get coherent variation
            timestep = torch.tensor([int(self.config.creativity_temperature * 
                                       self.diffusion_core.config.num_timesteps)], 
                                  device=self.diffusion_core.device)
            
            variation_embedding = self.diffusion_core.denoise_step(
                creative_embedding, timestep, prompt_embedding
            )
            
            # Convert to text variation
            variation_text = self._embedding_to_idea_variation(variation_embedding, prompt, i)
            variations.append(variation_text)
            
        return variations
    
    def _embedding_to_idea_variation(self, embedding: torch.Tensor, original_prompt: str, variation_id: int) -> str:
        """Convert embedding to idea variation text"""
        # This is a simplified approach - in practice, use a trained decoder
        
        # Create variations by modifying the original prompt
        variation_templates = [
            "What if we approached {prompt} from a {angle} perspective?",
            "How might {prompt} be different if we considered {factor}?",
            "Could {prompt} be enhanced by incorporating {element}?",
            "What are the implications of {prompt} for {domain}?",
            "How does {prompt} relate to {connection}?"
        ]
        
        # Use embedding values to select template and fill parameters
        embedding_np = embedding.cpu().numpy().flatten()
        template_idx = int(abs(embedding_np[0]) * len(variation_templates)) % len(variation_templates)
        template = variation_templates[template_idx]
        
        # Generate creative parameters based on embedding
        angles = ["interdisciplinary", "historical", "futuristic", "practical", "theoretical"]
        factors = ["cultural context", "technological constraints", "ethical implications", "economic factors"]
        elements = ["machine learning", "human psychology", "systems thinking", "data visualization"]
        domains = ["education", "healthcare", "sustainability", "social justice", "innovation"]
        connections = ["emerging technologies", "social trends", "scientific discoveries", "policy changes"]
        
        # Select parameters based on embedding values
        angle = angles[int(abs(embedding_np[1]) * len(angles)) % len(angles)]
        factor = factors[int(abs(embedding_np[2]) * len(factors)) % len(factors)]
        element = elements[int(abs(embedding_np[3]) * len(elements)) % len(elements)]
        domain = domains[int(abs(embedding_np[4]) * len(domains)) % len(domains)]
        connection = connections[int(abs(embedding_np[5]) * len(connections)) % len(connections)]
        
        # Format the variation
        variation = template.format(
            prompt=original_prompt,
            angle=angle,
            factor=factor,
            element=element,
            domain=domain,
            connection=connection
        )
        
        return variation
    
    def _expand_idea_recursively(self, idea: str, depth: int) -> Dict[str, Any]:
        """Recursively expand an idea to specified depth"""
        if depth <= 0:
            return {'idea': idea, 'expansions': []}
            
        # Generate sub-ideas
        sub_ideas = self._generate_sub_ideas(idea)
        
        # Recursively expand sub-ideas
        expanded_sub_ideas = []
        for sub_idea in sub_ideas:
            expansion = self._expand_idea_recursively(sub_idea, depth - 1)
            expanded_sub_ideas.append(expansion)
            
        return {
            'idea': idea,
            'expansions': expanded_sub_ideas,
            'depth': depth,
            'creativity_score': self._calculate_creativity_score(idea)
        }
    
    def _generate_sub_ideas(self, parent_idea: str) -> List[str]:
        """Generate sub-ideas from a parent idea"""
        # Encode parent idea
        parent_embedding = self.diffusion_core.encode_text([parent_idea])
        
        sub_ideas = []
        num_sub_ideas = 3  # Generate 3 sub-ideas per parent
        
        for i in range(num_sub_ideas):
            # Add focused noise for sub-idea generation
            sub_noise = torch.randn_like(parent_embedding) * (self.config.creativity_temperature * 0.5)
            sub_embedding = parent_embedding + sub_noise
            
            # Generate sub-idea text
            sub_idea = self._embedding_to_sub_idea(sub_embedding, parent_idea, i)
            sub_ideas.append(sub_idea)
            
        return sub_ideas
    
    def _embedding_to_sub_idea(self, embedding: torch.Tensor, parent_idea: str, sub_id: int) -> str:
        """Convert embedding to sub-idea text"""
        sub_idea_templates = [
            "Specifically, we could investigate {aspect} of {parent}",
            "This leads to the question: {question} about {parent}",
            "A practical application might be {application} based on {parent}"
        ]
        
        # Use embedding to select template and parameters
        embedding_np = embedding.cpu().numpy().flatten()
        template_idx = sub_id % len(sub_idea_templates)
        template = sub_idea_templates[template_idx]
        
        # Generate parameters
        aspects = ["the underlying mechanisms", "the practical implications", "the theoretical foundations"]
        questions = ["How can we measure", "What factors influence", "How might we optimize"]
        applications = ["a tool or system", "a methodology", "an intervention"]
        
        aspect = aspects[int(abs(embedding_np[0]) * len(aspects)) % len(aspects)]
        question = questions[int(abs(embedding_np[1]) * len(questions)) % len(questions)]
        application = applications[int(abs(embedding_np[2]) * len(applications)) % len(applications)]
        
        return template.format(
            aspect=aspect,
            question=question,
            application=application,
            parent=parent_idea
        )
    
    def _synthesize_novel_angles(self, expanded_ideas: List[Dict[str, Any]], original_prompt: str) -> List[str]:
        """Synthesize novel angles by combining different idea expansions"""
        novel_angles = []
        
        # Extract all ideas from expansions
        all_ideas = []
        for expansion in expanded_ideas:
            all_ideas.extend(self._extract_all_ideas_from_expansion(expansion))
            
        # Generate novel combinations
        for i in range(min(5, len(all_ideas))):  # Generate up to 5 novel angles
            # Select random ideas to combine
            selected_ideas = np.random.choice(all_ideas, size=min(3, len(all_ideas)), replace=False)
            
            # Create novel angle by synthesis
            novel_angle = self._synthesize_ideas(selected_ideas.tolist(), original_prompt)
            novel_angles.append(novel_angle)
            
        return novel_angles
    
    def _extract_all_ideas_from_expansion(self, expansion: Dict[str, Any]) -> List[str]:
        """Extract all ideas from an expansion tree"""
        ideas = [expansion['idea']]
        
        for sub_expansion in expansion.get('expansions', []):
            ideas.extend(self._extract_all_ideas_from_expansion(sub_expansion))
            
        return ideas
    
    def _synthesize_ideas(self, ideas: List[str], original_prompt: str) -> str:
        """Synthesize multiple ideas into a novel angle"""
        # Simple synthesis approach
        synthesis_templates = [
            "What if we combined {idea1} with {idea2} to address {original}?",
            "By integrating insights from {idea1} and {idea2}, we might discover new approaches to {original}",
            "The intersection of {idea1} and {idea2} could reveal unexplored aspects of {original}"
        ]
        
        template = np.random.choice(synthesis_templates)
        
        return template.format(
            idea1=ideas[0] if len(ideas) > 0 else "this concept",
            idea2=ideas[1] if len(ideas) > 1 else "related ideas",
            original=original_prompt
        )
    
    def _calculate_creativity_score(self, idea: str) -> float:
        """Calculate creativity score for an idea"""
        # Simple creativity scoring based on text characteristics
        words = idea.lower().split()
        
        # Factors that might indicate creativity
        creative_words = ['novel', 'innovative', 'unique', 'creative', 'original', 'breakthrough']
        question_words = ['what', 'how', 'why', 'could', 'might', 'if']
        
        creative_word_count = sum(1 for word in words if word in creative_words)
        question_word_count = sum(1 for word in words if word in question_words)
        
        # Normalize scores
        creativity_score = (creative_word_count * 0.3 + question_word_count * 0.2 + 
                          min(1.0, len(words) / 20) * 0.5)
        
        return min(1.0, creativity_score)
    
    def _rank_ideas_by_creativity(self, ideas: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Rank ideas by creativity and feasibility scores"""
        ranked_ideas = []
        
        for i, idea in enumerate(ideas):
            if isinstance(idea, str):
                idea_text = idea
                creativity_score = self._calculate_creativity_score(idea_text)
            else:
                idea_text = idea.get('idea', '')
                creativity_score = idea.get('creativity_score', self._calculate_creativity_score(idea_text))
                
            feasibility_score = self._calculate_feasibility_score(idea_text)
            
            ranked_ideas.append({
                'idea': idea_text,
                'creativity_score': creativity_score,
                'feasibility_score': feasibility_score,
                'composite_score': creativity_score * 0.6 + feasibility_score * 0.4,
                'rank': i + 1
            })
            
        # Sort by composite score
        ranked_ideas.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Update ranks
        for i, idea in enumerate(ranked_ideas):
            idea['rank'] = i + 1
            
        return ranked_ideas
    
    def _calculate_feasibility_score(self, idea: str) -> float:
        """Calculate feasibility score for an idea"""
        words = idea.lower().split()
        
        # Factors that might indicate feasibility
        practical_words = ['practical', 'implement', 'apply', 'use', 'tool', 'method', 'approach']
        complex_words = ['complex', 'difficult', 'challenging', 'impossible', 'theoretical']
        
        practical_count = sum(1 for word in words if word in practical_words)
        complex_count = sum(1 for word in words if word in complex_words)
        
        # Higher practical words increase feasibility, complex words decrease it
        feasibility_score = (practical_count * 0.3 - complex_count * 0.2 + 0.5)
        
        return max(0.0, min(1.0, feasibility_score))