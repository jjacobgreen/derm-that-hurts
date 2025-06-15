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

from model import MONETInference
from data import load_isic_dataset, get_comprehensive_dermatology_concepts
from utils import analyze_prediction_confidence

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