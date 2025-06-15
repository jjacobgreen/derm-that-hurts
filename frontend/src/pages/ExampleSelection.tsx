
import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { ErrorMessage } from '../components/shared/ErrorMessage';
import { Button } from '../components/shared/Button';

export const ExampleSelection: React.FC = () => {
  const { 
    exampleImages, 
    loading, 
    error, 
    setExampleImages, 
    setLoading, 
    setError, 
    setSelectedImage, 
    setCurrentPage 
  } = useAppStore();

  useEffect(() => {
    const fetchExamples = async () => {
      try {
        setLoading(true);
        setError(undefined);
        const response = await apiService.getRandomExamples();
        setExampleImages(response.images);
      } catch (err) {
        console.error('Failed to fetch examples:', err);
        setError('failed to load example images. please try again.');
      } finally {
        setLoading(false);
      }
    };

    if (!exampleImages) {
      fetchExamples();
    }
  }, [exampleImages, setExampleImages, setLoading, setError]);

  const handleImageSelect = (image: any, index: number) => {
    setSelectedImage({
      type: 'example',
      data: image,
      index: image.index,
      url: image.image // base64 encoded image
    });
    setCurrentPage(4);
  };

  const handleBack = () => {
    setCurrentPage(1);
  };

  const handleRetry = () => {
    setExampleImages(undefined);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center gradient-border">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-soft-grey lowercase">loading example images...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 gradient-border">
        <ErrorMessage 
          message={error} 
          onRetry={handleRetry}
          onBack={handleBack}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen py-12 px-4 gradient-border">
      <div className="max-w-4xl mx-auto animate-fade-in relative z-10">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-4 lowercase">
            choose an example image
          </h1>
          <p className="text-soft-grey text-lg lowercase">
            select one of these dermatology images to analyze
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {exampleImages?.map((image, index) => (
            <div
              key={image.index || index}
              className="image-grid-item bg-white border-2 border-transparent hover:border-lilac/30"
              onClick={() => handleImageSelect(image, index)}
            >
              <img
                src={image.image}
                alt={`example dermatology image ${index + 1}`}
                className="w-full h-full object-cover"
                loading="lazy"
              />
            </div>
          ))}
        </div>

        <div className="text-center">
          <Button variant="cancel" onClick={handleBack}>
            back to home
          </Button>
        </div>
      </div>
    </div>
  );
};
