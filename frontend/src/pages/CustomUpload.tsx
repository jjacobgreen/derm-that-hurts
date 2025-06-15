
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { Button } from '../components/shared/Button';
import { ErrorMessage } from '../components/shared/ErrorMessage';

export const CustomUpload: React.FC = () => {
  const { setSelectedImage, setCurrentPage, setError, error } = useAppStore();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, or JPEG).');
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB
      setError('File size must be under 10MB.');
      return;
    }

    try {
      setUploading(true);
      setError(undefined);
      setUploadProgress(0);

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 100);

      const result = await apiService.uploadImage(file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);

      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      
      setSelectedImage({
        type: 'upload',
        data: file,
        uploadId: result.upload_id,
        url: previewUrl
      });

      setTimeout(() => {
        setCurrentPage(4);
      }, 500);

    } catch (err) {
      console.error('Upload failed:', err);
      setError('Sorry, we couldn\'t upload your image. Please try again.');
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  }, [setSelectedImage, setCurrentPage, setError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1,
    disabled: uploading
  });

  const handleBack = () => {
    setCurrentPage(1);
    setError(undefined);
  };

  const handleRetry = () => {
    setError(undefined);
    setUploadProgress(0);
  };

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <ErrorMessage 
          message={error} 
          onRetry={handleRetry}
          onBack={handleBack}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-2xl mx-auto animate-fade-in">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Upload Your Image
          </h1>
          <p className="text-soft-grey text-lg">
            Upload a dermatology image for AI analysis
          </p>
        </div>

        <div className="derm-card mb-8">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${
              isDragActive 
                ? 'border-lilac bg-lilac/5' 
                : uploading 
                ? 'border-gray-300 bg-gray-50 cursor-not-allowed'
                : 'border-gray-300 hover:border-lilac hover:bg-lilac/5'
            }`}
          >
            <input {...getInputProps()} />
            
            {uploading ? (
              <div className="space-y-4">
                <div className="text-4xl">📤</div>
                <div>
                  <p className="text-lg font-medium text-gray-700 mb-2">
                    Uploading...
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-lilac h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-sm text-soft-grey mt-1">
                    {uploadProgress}% complete
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-6xl">📋</div>
                <div>
                  <p className="text-xl font-medium text-gray-700 mb-2">
                    {isDragActive 
                      ? 'Drop your image here...' 
                      : 'Drag & drop your image here'
                    }
                  </p>
                  <p className="text-soft-grey mb-4">or</p>
                  <div className="inline-block">
                    <span className="bg-lilac text-white px-6 py-2 rounded-lg font-medium">
                      Choose File
                    </span>
                  </div>
                </div>
                <div className="text-sm text-soft-grey">
                  <p>Accepted formats: JPG, PNG, JPEG</p>
                  <p>Maximum file size: 10MB</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="text-center">
          <Button variant="cancel" onClick={handleBack} disabled={uploading}>
            Back to Home
          </Button>
        </div>
      </div>
    </div>
  );
};
