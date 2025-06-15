
import React from 'react';
import { useAppStore } from '../store/appStore';
import { Button } from '../components/shared/Button';

export const ImagePreview: React.FC = () => {
  const { selectedImage, setCurrentPage } = useAppStore();

  if (!selectedImage) {
    setCurrentPage(1);
    return null;
  }

  const handleAnalyze = () => {
    setCurrentPage(5);
  };

  const handleCancel = () => {
    setCurrentPage(1);
  };

  return (
    <div className="min-h-screen py-12 px-4 gradient-border">
      <div className="max-w-2xl mx-auto animate-fade-in relative z-10">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Confirm Your Image
          </h1>
          <p className="text-soft-grey text-lg">
            Review your image before analysis
          </p>
        </div>

        <div className="derm-card text-center mb-8">
          <div className="mb-6">
            <div className="inline-block border-4 border-gray-100 rounded-3xl overflow-hidden">
              <img
                src={selectedImage.url}
                alt={selectedImage.type === 'example' ? `Example image ${selectedImage.index}` : 'Your uploaded image'}
                className="w-56 h-56 object-cover"
              />
            </div>
          </div>
          
          <p className="text-soft-grey mb-6">
            {selectedImage.type === 'example' 
              ? `Example image #${(selectedImage.index || 0) + 1}` 
              : 'Your uploaded image'
            }
          </p>
          
          <div className="text-sm text-soft-grey bg-gray-50 rounded-2xl p-4">
            <p className="font-medium mb-1">Analysis Details:</p>
            <p>• Image will be resized to 224x224 pixels for processing</p>
            <p>• Analysis typically takes 30-60 seconds</p>
            <p>• Results will show confidence scores for detected concepts</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md mx-auto">
          <Button 
            variant="cancel" 
            onClick={handleCancel}
            className="w-full"
          >
            Cancel
          </Button>
          
          <Button 
            variant="primary" 
            onClick={handleAnalyze}
            className="w-full"
          >
            Analyze Image
          </Button>
        </div>
      </div>
    </div>
  );
};
