import { ExampleImage, AnalysisResult, UploadResponse } from '../types/app';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} - ${errorText}`);
    }
    return response.json();
  }

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    console.log('Checking API health...');
    const response = await fetch(`${API_BASE_URL}/health`);
    return this.handleResponse(response);
  }

  async getModelStatus(): Promise<{ loaded: boolean; device: string; model_name: string }> {
    console.log('Checking model status...');
    const response = await fetch(`${API_BASE_URL}/model/status`);
    return this.handleResponse(response);
  }

  async getRandomExamples(count: number = 5): Promise<{ images: ExampleImage[] }> {
    console.log(`Fetching ${count} random example images...`);
    const response = await fetch(`${API_BASE_URL}/dataset/random/${count}`);
    return this.handleResponse(response);
  }

  async getDatasetImage(index: number): Promise<ExampleImage> {
    console.log('Fetching dataset image...', index);
    const response = await fetch(`${API_BASE_URL}/dataset/image/${index}`);
    return this.handleResponse(response);
  }

  async uploadImage(file: File): Promise<UploadResponse> {
    console.log('Uploading image...', file.name);
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    return this.handleResponse(response);
  }

  async getUploadedImage(uploadId: string): Promise<{ upload_id: string; image: string; size: number[]; mode: string }> {
    console.log('Fetching uploaded image...', uploadId);
    const response = await fetch(`${API_BASE_URL}/upload/${uploadId}`);
    return this.handleResponse(response);
  }

  async analyzeExample(index: number, options?: { concepts?: string[]; threshold?: number }): Promise<{ job_id: string; status: string }> {
    console.log('Analyzing example image...', index);
    const response = await fetch(`${API_BASE_URL}/analyze/dataset/${index}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options || {}),
    });
    
    return this.handleResponse(response);
  }

  async analyzeUpload(uploadId: string, options?: { concepts?: string[]; threshold?: number }): Promise<{ job_id: string; status: string }> {
    console.log('Analyzing uploaded image...', uploadId);
    const response = await fetch(`${API_BASE_URL}/analyze/upload/${uploadId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options || {}),
    });
    
    return this.handleResponse(response);
  }

  async getAnalysisResult(jobId: string): Promise<AnalysisResult> {
    console.log('Checking analysis result...', jobId);
    const response = await fetch(`${API_BASE_URL}/analyze/result/${jobId}`);
    return this.handleResponse(response);
  }

  async getResultsSummary(jobId: string): Promise<any> {
    console.log('Fetching results summary...', jobId);
    const response = await fetch(`${API_BASE_URL}/results/${jobId}/summary`);
    return this.handleResponse(response);
  }

  async getResultsVisualization(jobId: string): Promise<{ chart_data: any }> {
    console.log('Fetching results visualization...', jobId);
    const response = await fetch(`${API_BASE_URL}/results/${jobId}/visualization`);
    return this.handleResponse(response);
  }

  async getConcepts(): Promise<{ concepts: string[] }> {
    console.log('Fetching available concepts...');
    const response = await fetch(`${API_BASE_URL}/concepts`);
    return this.handleResponse(response);
  }

  async pollAnalysisResult(jobId: string, maxAttempts: number = 30): Promise<AnalysisResult> {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      
      const poll = async () => {
        try {
          attempts++;
          const result = await this.getAnalysisResult(jobId);
          
          if (result.status === 'completed') {
            resolve(result);
          } else if (result.status === 'failed') {
            reject(new Error(result.error || 'Analysis failed'));
          } else if (attempts >= maxAttempts) {
            reject(new Error('Analysis timeout - maximum attempts reached'));
          } else {
            // Continue polling every 2 seconds
            setTimeout(poll, 2000);
          }
        } catch (error) {
          if (attempts >= maxAttempts) {
            reject(error);
          } else {
            // Retry on network errors
            setTimeout(poll, 2000);
          }
        }
      };
      
      poll();
    });
  }
}

export const apiService = new ApiService();
