from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Annotated
import uuid
import io
import base64
import numpy as np
from PIL import Image
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

import os
import sys

# Add the src directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from model import MONETInference
# from data import load_isic_dataset, get_comprehensive_dermatology_concepts
# from utils import analyze_prediction_confidence

# data.py
from datasets import load_dataset


def load_isic_dataset(split="train", num_samples=None):
    """
    Load ISIC 2019 dataset from Hugging Face.
    
    Args:
        split (str): Dataset split to load ("train", "validation", "test")
        num_samples (int): Number of samples to load (None for all)
        
    Returns:
        Dataset: Loaded ISIC dataset
    """
    print(f"Loading ISIC 2019 dataset ({split} split)...")
    
    try:
        dataset = load_dataset("MKZuziak/ISIC_2019_224", split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        print(f"Loaded {len(dataset)} samples from ISIC dataset")
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and the datasets library installed")
        return None


def get_comprehensive_dermatology_concepts():
    """
    Get a comprehensive list of dermatology concepts for annotation.
    
    Returns:
        list: List of dermatology concepts
    """
    return [
        # Primary skin cancers
        "melanoma",
        "basal cell carcinoma", 
        "squamous cell carcinoma",
        
        # Precancerous lesions
        "actinic keratosis",
        "Bowen's disease",
        "atypical nevus",
        
        # Benign lesions
        "seborrheic keratosis",
        "dermatofibroma",
        "nevus",
        "solar lentigo",
        "cherry angioma",
        
        # Vascular lesions
        "vascular lesion",
        "hemangioma",
        "port wine stain",
        
        # Morphological features
        "pigmented lesion",
        "asymmetric lesion",
        "irregular border",
        "color variation",
        "ulceration",
        "scaling",
        "crusting",
        
        # Inflammatory changes
        "inflammatory changes",
        "erythema",
        "papule",
        "nodule",
        
        # Infectious conditions
        "bacterial infection",
        "fungal infection",
        "viral wart",
        
        # Other conditions
        "eczema",
        "psoriasis",
        "contact dermatitis"
    ]

# model.py
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

# utils.py
import numpy as np


def analyze_prediction_confidence(results, threshold=0.1):
    """
    Analyze prediction confidence and provide insights.
    
    Args:
        results (dict): Results from concept annotation
        threshold (float): Minimum score threshold for high confidence
        
    Returns:
        dict: Analysis results
    """
    scores = [item['score'] for item in results['ranked_concepts']]
    
    analysis = {
        'max_score': max(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'high_confidence_count': len([s for s in scores if s > threshold]),
        'top_3_concepts': results['ranked_concepts'][:3]
    }
    
    return analysis

# main.py

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ModelStatus(BaseModel):
    loaded: bool
    device: str
    model_name: str

class DatasetStatus(BaseModel):
    loaded: bool
    total_samples: int

class ImageInfo(BaseModel):
    index: int
    size: tuple
    mode: str
    metadata: Dict[str, Any]

class AnalysisRequest(BaseModel):
    concepts: Optional[List[str]] = None
    threshold: Optional[float] = 0.1

class AnalysisResult(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    max_score: Optional[float] = None
    mean_score: Optional[float] = None
    high_confidence_count: Optional[int] = None
    top_concepts: Optional[List[Dict]] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    size: tuple
    mode: str

# Dependency classes for state management
class ModelManager:
    def __init__(self):
        self._model: Optional[MONETInference] = None
        self._loading = False
    
    async def get_model(self) -> MONETInference:
        if self._model is None and not self._loading:
            self._loading = True
            print("Loading MONET model...")
            self._model = MONETInference(device="auto")
            self._loading = False
            print("MONET model loaded!")
        
        while self._loading:
            await asyncio.sleep(0.1)
        
        if self._model is None:
            raise HTTPException(status_code=503, detail="Model failed to load")
        
        return self._model
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def get_device(self) -> str:
        return self._model.device if self._model else "unknown"

class DatasetManager:
    def __init__(self):
        self._dataset = None
        self._loading = False
    
    async def get_dataset(self):
        if self._dataset is None and not self._loading:
            self._loading = True
            print("Loading ISIC dataset...")
            self._dataset = load_isic_dataset(split="train", num_samples=500)
            self._loading = False
            print(f"Dataset loaded: {len(self._dataset) if self._dataset else 0} samples")
        
        while self._loading:
            await asyncio.sleep(0.1)
        
        if self._dataset is None:
            raise HTTPException(status_code=503, detail="Dataset failed to load")
        
        return self._dataset
    
    def is_loaded(self) -> bool:
        return self._dataset is not None
    
    def get_sample_count(self) -> int:
        return len(self._dataset) if self._dataset else 0

class UploadManager:
    def __init__(self):
        self._uploads: Dict[str, Image.Image] = {}
    
    def store_image(self, image: Image.Image) -> str:
        upload_id = str(uuid.uuid4())
        self._uploads[upload_id] = image
        return upload_id
    
    def get_image(self, upload_id: str) -> Image.Image:
        if upload_id not in self._uploads:
            raise HTTPException(status_code=404, detail="Upload not found")
        return self._uploads[upload_id]
    
    def delete_image(self, upload_id: str) -> bool:
        if upload_id not in self._uploads:
            return False
        del self._uploads[upload_id]
        return True
    
    def has_image(self, upload_id: str) -> bool:
        return upload_id in self._uploads

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
    
    def create_job(self, image_type: str, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "image_type": image_type,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }
        return job_id
    
    def update_job(self, job_id: str, **updates):
        if job_id not in self._jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        self._jobs[job_id].update(updates)
    
    def get_job(self, job_id: str) -> Dict:
        if job_id not in self._jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return self._jobs[job_id]
    
    def job_exists(self, job_id: str) -> bool:
        return job_id in self._jobs

# Initialize managers
model_manager = ModelManager()
dataset_manager = DatasetManager()
upload_manager = UploadManager()
job_manager = JobManager()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model and dataset
    print("Starting up API...")
    await model_manager.get_model()
    await dataset_manager.get_dataset()
    print("API ready!")
    
    yield
    
    # Shutdown: Cleanup if needed
    print("Shutting down API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MONET Dermatology Analyzer API",
    description="AI-powered dermatology image analysis using MONET",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency functions
async def get_model() -> MONETInference:
    return await model_manager.get_model()

async def get_dataset():
    return await dataset_manager.get_dataset()

def get_upload_manager() -> UploadManager:
    return upload_manager

def get_job_manager() -> JobManager:
    return job_manager

# Type aliases for dependencies
ModelDep = Annotated[MONETInference, Depends(get_model)]
DatasetDep = Annotated[Any, Depends(get_dataset)]
UploadDep = Annotated[UploadManager, Depends(get_upload_manager)]
JobDep = Annotated[JobManager, Depends(get_job_manager)]

# Health & Status endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy", 
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    return ModelStatus(
        loaded=model_manager.is_loaded(),
        device=model_manager.get_device(),
        model_name="MONET" if model_manager.is_loaded() else "none"
    )

@app.get("/dataset/status", response_model=DatasetStatus)
async def get_dataset_status():
    return DatasetStatus(
        loaded=dataset_manager.is_loaded(),
        total_samples=dataset_manager.get_sample_count()
    )

# Dataset endpoints
@app.get("/dataset/random/{count}")
async def get_random_images(count: int, dataset: DatasetDep):
    if count > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images allowed")
    
    # Get random indices
    indices = np.random.choice(len(dataset), size=min(count, len(dataset)), replace=False)
    
    images_data = []
    for idx in indices:
        sample = dataset[int(idx)]
        
        # Convert PIL image to base64
        img_buffer = io.BytesIO()
        sample['image'].save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        images_data.append({
            "index": int(idx),
            "image": f"data:image/png;base64,{img_base64}",
            "size": sample['image'].size,
            "mode": sample['image'].mode,
            "metadata": {k: v for k, v in sample.items() if k != 'image'}
        })
    
    return {"images": images_data}

@app.get("/dataset/image/{index}")
async def get_dataset_image(index: int, dataset: DatasetDep):
    if index >= len(dataset) or index < 0:
        raise HTTPException(status_code=404, detail="Image index not found")
    
    sample = dataset[index]
    
    # Convert PIL image to base64
    img_buffer = io.BytesIO()
    sample['image'].save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return {
        "index": index,
        "image": f"data:image/png;base64,{img_base64}",
        "size": sample['image'].size,
        "mode": sample['image'].mode,
        "metadata": {k: v for k, v in sample.items() if k != 'image'}
    }

# Upload endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile, upload_mgr: UploadDep):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read and process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Store image
    upload_id = upload_mgr.store_image(image)
    
    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename,
        size=image.size,
        mode=image.mode
    )

@app.get("/upload/{upload_id}")
async def get_uploaded_image(upload_id: str, upload_mgr: UploadDep):
    image = upload_mgr.get_image(upload_id)
    
    # Convert PIL image to base64
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return {
        "upload_id": upload_id,
        "image": f"data:image/png;base64,{img_base64}",
        "size": image.size,
        "mode": image.mode
    }

@app.delete("/upload/{upload_id}")
async def delete_uploaded_image(upload_id: str, upload_mgr: UploadDep):
    if not upload_mgr.delete_image(upload_id):
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return {"message": "Image deleted successfully"}

# Analysis endpoints
async def run_analysis_task(
    job_id: str, 
    image: Image.Image, 
    concepts: List[str], 
    threshold: float,
    model: MONETInference,
    job_mgr: JobManager
):
    """Background task for running analysis."""
    try:
        # Update job status
        job_mgr.update_job(job_id, status="processing")
        
        # Run analysis
        results = model.annotate_concepts(image, concepts)
        analysis = analyze_prediction_confidence(results, threshold=threshold)
        
        # Store results
        job_mgr.update_job(job_id, 
            status="completed",
            results=results,
            analysis=analysis,
            max_score=analysis['max_score'],
            mean_score=analysis['mean_score'],
            high_confidence_count=analysis['high_confidence_count'],
            top_concepts=results['ranked_concepts'][:10]
        )
        
    except Exception as e:
        job_mgr.update_job(job_id, status="failed", error=str(e))

@app.post("/analyze/dataset/{index}")
async def analyze_dataset_image(
    index: int, 
    background_tasks: BackgroundTasks,
    dataset: DatasetDep,
    model: ModelDep,
    job_mgr: JobDep,
    request: AnalysisRequest = AnalysisRequest()
):
    if index >= len(dataset) or index < 0:
        raise HTTPException(status_code=404, detail="Image index not found")
    
    # Get image
    sample = dataset[index]
    image = sample['image']
    
    # Get concepts
    concepts = request.concepts or get_comprehensive_dermatology_concepts()
    
    # Create job
    job_id = job_mgr.create_job("dataset", image_index=index)
    
    # Start background task
    background_tasks.add_task(run_analysis_task, job_id, image, concepts, request.threshold, model, job_mgr)
    
    return {"job_id": job_id, "status": "queued"}

@app.post("/analyze/upload/{upload_id}")
async def analyze_uploaded_image(
    upload_id: str, 
    background_tasks: BackgroundTasks,
    upload_mgr: UploadDep,
    model: ModelDep,
    job_mgr: JobDep,
    request: AnalysisRequest = AnalysisRequest()
):
    # Get image
    image = upload_mgr.get_image(upload_id)
    
    # Get concepts
    concepts = request.concepts or get_comprehensive_dermatology_concepts()
    
    # Create job
    job_id = job_mgr.create_job("upload", upload_id=upload_id)
    
    # Start background task
    background_tasks.add_task(run_analysis_task, job_id, image, concepts, request.threshold, model, job_mgr)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/analyze/result/{job_id}", response_model=AnalysisResult)
async def get_analysis_result(job_id: str, job_mgr: JobDep):
    job_data = job_mgr.get_job(job_id)
    
    return AnalysisResult(
        job_id=job_id,
        status=job_data["status"],
        max_score=job_data.get("max_score"),
        mean_score=job_data.get("mean_score"),
        high_confidence_count=job_data.get("high_confidence_count"),
        top_concepts=job_data.get("top_concepts"),
        error=job_data.get("error")
    )

# Concepts endpoint
@app.get("/concepts")
async def get_concepts():
    return {"concepts": get_comprehensive_dermatology_concepts()}

# Results endpoints
@app.get("/results/{job_id}/summary")
async def get_results_summary(job_id: str, job_mgr: JobDep):
    job_data = job_mgr.get_job(job_id)
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    return {
        "job_id": job_id,
        "analysis": job_data["analysis"],
        "summary": {
            "max_score": job_data["max_score"],
            "mean_score": job_data["mean_score"],
            "high_confidence_count": job_data["high_confidence_count"]
        }
    }

@app.get("/results/{job_id}/visualization")
async def get_results_visualization(job_id: str, job_mgr: JobDep):
    job_data = job_mgr.get_job(job_id)
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    # Prepare data for frontend charting
    top_concepts = job_data["top_concepts"][:10]
    
    chart_data = {
        "labels": [item["concept"] for item in top_concepts],
        "scores": [item["score"] for item in top_concepts],
        "colors": ["#1f77b4" if item["score"] > 0.1 else "#ff7f0e" if item["score"] > 0.05 else "#d62728" for item in top_concepts]
    }
    
    return {"chart_data": chart_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 