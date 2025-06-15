export interface AppState {
  currentPage: 1 | 2 | 3 | 4 | 5 | 6;
  selectedImage?: {
    type: 'example' | 'upload';
    data: any;
    uploadId?: string;
    index?: number;
    url?: string;
  };
  analysisResult?: AnalysisResult;
  loading: boolean;
  error?: string;
  exampleImages?: ExampleImage[];
}

export interface ExampleImage {
  index: number;
  image: string; // base64 encoded image
  size: number[]; // [width, height]
  mode: string;
  metadata: Record<string, any>;
}

export interface AnalysisResult {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  max_score?: number;
  mean_score?: number;
  high_confidence_count?: number;
  top_concepts?: Concept[];
  error?: string;
}

export interface Concept {
  concept: string;
  score: number;
  rank: number;
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  size: number[]; // [width, height]
  mode: string;
}

export interface ModelStatus {
  loaded: boolean;
  device: string;
  model_name: string;
}

export interface DatasetStatus {
  loaded: boolean;
  total_samples: number;
}

export interface ConceptsResponse {
  concepts: string[];
}

export interface ChartData {
  labels: string[];
  scores: number[];
  colors: string[];
}
