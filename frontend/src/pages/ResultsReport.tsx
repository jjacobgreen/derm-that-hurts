import React, { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { Button } from '../components/shared/Button';

export const ResultsReport: React.FC = () => {
  const { selectedImage, analysisResult, setCurrentPage, resetState } = useAppStore();

  useEffect(() => {
    if (!selectedImage || !analysisResult || analysisResult.status !== 'completed') {
      setCurrentPage(1);
    }
  }, [selectedImage, analysisResult, setCurrentPage]);

  if (!selectedImage || !analysisResult || analysisResult.status !== 'completed') {
    return null;
  }

  const topConcepts = analysisResult.top_concepts
    ?.sort((a, b) => b.score - a.score)
    .slice(0, 10) || [];

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'bg-soft-green';
    if (confidence >= 0.4) return 'bg-soft-orange';
    return 'bg-gray-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.4) return 'Medium';
    return 'Low';
  };

  const handleAnalyzeAnother = () => {
    resetState();
    setCurrentPage(1);
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-4xl mx-auto animate-fade-in relative z-10">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-6 mb-6">
            <div className="w-32 h-32 border-4 border-gray-100 rounded-xl overflow-hidden">
              <img
                src={selectedImage.url}
                alt="Analyzed image"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="text-left">
              <h1 className="text-4xl font-bold text-gray-800 mb-2">
                Analysis Results
              </h1>
              <p className="text-soft-grey">
                {selectedImage.type === 'example' 
                  ? `Example image #${(selectedImage.index || 0) + 1}` 
                  : 'Your uploaded image'
                }
              </p>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="derm-card text-center">
            <div className="text-3xl font-bold text-lilac mb-2">
              {((analysisResult.max_score || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-soft-grey font-medium">Max Confidence</div>
          </div>
          
          <div className="derm-card text-center">
            <div className="text-3xl font-bold text-baby-blue mb-2">
              {analysisResult.high_confidence_count || 0}
            </div>
            <div className="text-soft-grey font-medium">High Confidence Concepts</div>
          </div>
          
          <div className="derm-card text-center">
            <div className="text-3xl font-bold text-soft-grey mb-2">
              {((analysisResult.mean_score || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-soft-grey font-medium">Average Score</div>
          </div>
        </div>

        {/* Top Concepts */}
        <div className="derm-card mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">
            Top Detected Concepts
          </h2>
          
          <div className="space-y-4">
            {topConcepts.map((concept, index) => (
              <div key={index} className="flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-gray-800 truncate">
                      {concept.concept}
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        getConfidenceColor(concept.score).replace('bg-', 'bg-') + ' text-white'
                      }`}>
                        {getConfidenceLabel(concept.score)}
                      </span>
                      <span className="text-sm font-mono text-soft-grey">
                        {(concept.score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${getConfidenceColor(concept.score)}`}
                      style={{ width: `${concept.score * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="text-center">
          <Button 
            onClick={handleAnalyzeAnother}
            size="lg"
            className="px-8"
          >
            Analyze Another Image
          </Button>
        </div>
      </div>
    </div>
  );
};
