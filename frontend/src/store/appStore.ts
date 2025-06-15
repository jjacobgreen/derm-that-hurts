
import { create } from 'zustand';
import { AppState } from '../types/app';

interface AppStore extends AppState {
  setCurrentPage: (page: AppState['currentPage']) => void;
  setSelectedImage: (image: AppState['selectedImage']) => void;
  setAnalysisResult: (result: AppState['analysisResult']) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | undefined) => void;
  setExampleImages: (images: AppState['exampleImages']) => void;
  resetState: () => void;
}

const initialState: AppState = {
  currentPage: 1,
  loading: false,
};

export const useAppStore = create<AppStore>((set) => ({
  ...initialState,
  setCurrentPage: (page) => set({ currentPage: page }),
  setSelectedImage: (image) => set({ selectedImage: image }),
  setAnalysisResult: (result) => set({ analysisResult: result }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setExampleImages: (images) => set({ exampleImages: images }),
  resetState: () => set(initialState),
}));
