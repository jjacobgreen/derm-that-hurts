import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms as T

class MONETInference:
    def __init__(self, device="auto"):
        """
        Initialize MONET model for inference using Hugging Face.
        
        Args:
            device (str): Device to run inference on. "auto" selects best available.
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load MONET model from Hugging Face
        print("Loading MONET model from Hugging Face...")
        model_name = "suinleelab/monet"
        
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("MONET model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading MONET model: {e}")
            print("Note: You may need to request access to the model on Hugging Face")
            print("Falling back to standard CLIP model...")
            
            # Fallback to standard CLIP if MONET is not accessible
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model.to(self.device)
            self.model.eval()
            print("Standard CLIP model loaded as fallback")
    
    def annotate_concepts(self, image, concept_list, return_scores=True):
        """
        Annotate concepts in an image using MONET.
        
        Args:
            image (PIL.Image): Input image
            concept_list (list): List of medical concepts to check
            return_scores (bool): Whether to return similarity scores
            
        Returns:
            dict: Concept annotations and scores
        """
        # Prepare text prompts for concepts
        # MONET works better with medical-specific prompts
        text_prompts = [f"dermatology image showing {concept}" for concept in concept_list]
        
        # Process inputs
        inputs = self.processor(
            text=text_prompts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get similarity scores (logits per image)
            logits_per_image = outputs.logits_per_image
            similarity_scores = torch.softmax(logits_per_image, dim=-1)
            similarity_scores = similarity_scores.squeeze().cpu().numpy()
        
        # Handle single concept case
        if len(concept_list) == 1:
            similarity_scores = [similarity_scores.item()]
        
        # Create results dictionary
        results = {
            'concepts': concept_list,
            'scores': similarity_scores.tolist(),
            'ranked_concepts': []
        }
        
        # Rank concepts by similarity score
        ranked_indices = np.argsort(similarity_scores)[::-1]
        for idx in ranked_indices:
            results['ranked_concepts'].append({
                'concept': concept_list[idx],
                'score': float(similarity_scores[idx]),
                'rank': len(results['ranked_concepts']) + 1
            })
        
        return results
    
    def get_image_embedding(self, image):
        """
        Get image embedding from MONET.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Image embedding
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def get_text_embedding(self, text_list):
        """
        Get text embeddings from MONET.
        
        Args:
            text_list (list): List of text descriptions
            
        Returns:
            torch.Tensor: Text embeddings
        """
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            
        return text_features
    
    def compute_similarity_matrix(self, image, concept_list):
        """
        Compute similarity matrix between image and concepts.
        
        Args:
            image (PIL.Image): Input image
            concept_list (list): List of medical concepts
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        image_features = self.get_image_embedding(image)
        text_features = self.get_text_embedding(concept_list)
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(image_features, text_features.t())
        return similarity_matrix.cpu().numpy()
    
    def visualize_results(self, image, results, top_k=5):
        """
        Visualize concept annotation results.
        
        Args:
            image (PIL.Image): Original image
            results (dict): Results from annotate_concepts
            top_k (int): Number of top concepts to display
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        ax1.imshow(image)
        ax1.set_title("Input Dermatology Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Display top concepts
        top_concepts = results['ranked_concepts'][:top_k]
        concepts = [item['concept'] for item in top_concepts]
        scores = [item['score'] for item in top_concepts]
        
        # Create horizontal bar chart
        bars = ax2.barh(range(len(concepts)), scores)
        ax2.set_yticks(range(len(concepts)))
        ax2.set_yticklabels(concepts, fontsize=11)
        ax2.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'Top {top_k} MONET Concept Annotations', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # Color bars based on score (viridis colormap)
        max_score = max(scores) if scores else 1
        for i, bar in enumerate(bars):
            normalized_score = scores[i] / max_score
            bar.set_color(plt.cm.viridis(normalized_score))
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()