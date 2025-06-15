import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { ErrorMessage } from '../components/shared/ErrorMessage';

export const LoadingPage: React.FC = () => {
  const { 
    selectedImage, 
    setCurrentPage, 
    setAnalysisResult, 
    setError, 
    error 
  } = useAppStore();

  useEffect(() => {
    if (!selectedImage) {
      setCurrentPage(1);
      return;
    }

    const startAnalysis = async () => {
      try {
        setError(undefined);
        
        let jobResponse;
        if (selectedImage.type === 'example' && selectedImage.index !== undefined) {
          jobResponse = await apiService.analyzeExample(selectedImage.index);
        } else if (selectedImage.type === 'upload' && selectedImage.uploadId) {
          jobResponse = await apiService.analyzeUpload(selectedImage.uploadId);
        } else {
          throw new Error('Invalid image selection');
        }

        console.log('Analysis started, job ID:', jobResponse.job_id);
        
        // Poll for results
        const result = await apiService.pollAnalysisResult(jobResponse.job_id);
        console.log('Analysis completed:', result);
        
        setAnalysisResult(result);
        setCurrentPage(6);
        
      } catch (err) {
        console.error('Analysis failed:', err);
        setError('Analysis failed. Please try with a different image.');
      }
    };

    startAnalysis();
  }, [selectedImage, setCurrentPage, setAnalysisResult, setError]);

  const handleRetry = () => {
    setError(undefined);
    // The useEffect will automatically restart the analysis
  };

  const handleBack = () => {
    setCurrentPage(4);
    setError(undefined);
  };

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 gradient-border">
        <div className="relative z-10">
          <ErrorMessage 
            message={error} 
            onRetry={handleRetry}
            onBack={handleBack}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 gradient-border">
      <div className="text-center animate-fade-in relative z-10">
        <div className="derm-card max-w-md">
          <LoadingSpinner size="lg" />
          
          <h2 className="text-2xl font-bold text-gray-800 mt-6 mb-4">
            Analyzing Your Image...
          </h2>
          
          <p className="text-soft-grey mb-6">
            Our AI is processing your dermatology image using advanced MONET algorithms.
          </p>
          
          <div className="text-sm text-soft-grey bg-gray-50 rounded-2xl p-4">
            <p className="font-medium mb-2">What's happening:</p>
            <div className="space-y-1 text-left">
              <p>• Image preprocessing and normalization</p>
              <p>• Feature extraction using neural networks</p>
              <p>• Concept detection and confidence scoring</p>
              <p>• Results compilation and validation</p>
            </div>
          </div>
          
          <p className="text-xs text-soft-grey mt-4">
            This may take 30-60 seconds
          </p>
        </div>
      </div>
    </div>
  );
};
