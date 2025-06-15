# MONET Dermatology Analyzer

AI-powered dermatology image analysis using MONET (Medical cONcept annoTation) with a React frontend and FastAPI backend.

## 🏗️ Architecture

- **Backend**: FastAPI with MONET/CLIP model for image analysis
- **Frontend**: React + TypeScript with Vite and Tailwind CSS
- **Dataset**: ISIC 2019 dermatology images from Hugging Face
- **Infrastructure**: Docker Compose for easy deployment

## 🚀 Quick Start

### Option 1: Docker (Recommended)

The easiest way to run the entire application:

```bash
# Start all services with Docker
./start_docker.sh

# Or manually:
docker-compose up --build
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Manual Development Setup

Run the backend and frontend separately for development:

#### Backend Setup

```bash
# Start the FastAPI backend
./start_backend.sh

# Or manually:
cd app
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

In a new terminal:

```bash
# Start the React frontend
./start_frontend.sh

# Or manually:
cd frontend
npm install
npm run dev
```

## 📋 Features

### 🎯 Core Functionality

1. **Dataset Image Selection**: Browse and select from ISIC 2019 dataset
2. **Custom Image Upload**: Upload your own dermatology images
3. **AI Analysis**: MONET-powered concept annotation with confidence scores
4. **Interactive Results**: Visualize top concepts with confidence levels
5. **Progress Tracking**: Real-time analysis progress with status updates

### 🔧 API Endpoints

#### Health & Status
- `GET /health` - API health check
- `GET /model/status` - Model loading status
- `GET /dataset/status` - Dataset availability

#### Dataset Management  
- `GET /dataset/random/{count}` - Get random sample images
- `GET /dataset/image/{index}` - Get specific dataset image

#### Image Upload
- `POST /upload` - Upload image file
- `GET /upload/{upload_id}` - Retrieve uploaded image
- `DELETE /upload/{upload_id}` - Delete uploaded image

#### Analysis & Inference
- `POST /analyze/dataset/{index}` - Analyze dataset image
- `POST /analyze/upload/{upload_id}` - Analyze uploaded image
- `GET /analyze/result/{job_id}` - Get analysis results

#### Results & Visualization
- `GET /results/{job_id}/summary` - Get detailed analysis summary
- `GET /results/{job_id}/visualization` - Get chart visualization data
- `GET /concepts` - Get available dermatology concepts

## 🛠️ Development

### Project Structure

```
am-i-dying/
├── app/                    # FastAPI backend
│   ├── main_fastapi.py    # Main FastAPI application
│   ├── model.py           # MONET model wrapper
│   ├── data.py            # Dataset loading utilities
│   ├── utils.py           # Analysis utilities
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API service layer
│   │   ├── types/         # TypeScript type definitions
│   │   └── store/         # State management
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Frontend container
├── compose.yaml           # Docker Compose configuration
└── README.md             # This file
```

### Key Technologies

**Backend:**
- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- Transformers - Hugging Face model library
- CLIP/MONET - Vision-language models
- Uvicorn - ASGI server

**Frontend:**
- React 18 - UI framework
- TypeScript - Type safety
- Vite - Build tool and dev server
- Tailwind CSS - Styling framework
- Radix UI - Accessible components
- React Query - Data fetching
- Zustand - State management

## 🔬 Medical Concepts

The system analyzes images for 35+ dermatological concepts including:

**Primary Skin Cancers:**
- Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma

**Precancerous Lesions:**
- Actinic Keratosis, Bowen's Disease, Atypical Nevus

**Benign Lesions:**
- Seborrheic Keratosis, Dermatofibroma, Nevus, Solar Lentigo

**Morphological Features:**
- Asymmetric Lesion, Irregular Border, Color Variation, Ulceration

## 📊 Model Information

- **Model**: MONET (Medical cONcept annoTation)
- **Fallback**: OpenAI CLIP-ViT-Large if MONET unavailable
- **Dataset**: ISIC 2019 (500 samples loaded)
- **Concepts**: 35+ dermatology-specific terms
- **Confidence Threshold**: Configurable (default: 0.1)

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up --build -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop all services
docker-compose down

# Rebuild a specific service
docker-compose build backend
docker-compose up backend

# Access backend container
docker-compose exec backend bash
```

## 🌐 Environment Variables

### Backend
- `TOKENIZERS_PARALLELISM=false` - Disable tokenizer warnings
- `PYTHONWARNINGS=ignore` - Suppress Python warnings

### Frontend
- `VITE_API_BASE_URL` - Backend API URL (default: http://localhost:8000)

## 🚨 Troubleshooting

### Common Issues

**Backend won't start:**
- Ensure Python 3.8+ is installed
- Check if port 8000 is available
- Verify all dependencies are installed

**Frontend build fails:**
- Ensure Node.js 16+ is installed
- Delete `node_modules` and run `npm install`
- Check for TypeScript errors

**Model loading issues:**
- Requires internet connection for initial model download
- Models are cached after first download
- Check available disk space (models are ~500MB)

**Docker issues:**
- Ensure Docker and Docker Compose are installed
- Check if ports 3000 and 8000 are available
- Try `docker-compose down` and restart

### Logs and Debugging

```bash
# Backend logs
tail -f app.log

# Frontend logs  
npm run dev  # Shows build/runtime errors

# Docker logs
docker-compose logs -f [service-name]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and model licensing terms.

---

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified medical professionals for health concerns.
