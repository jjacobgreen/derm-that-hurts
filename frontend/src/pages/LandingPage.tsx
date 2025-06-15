
import React, { useState } from 'react';
import { Button } from '../components/shared/Button';
import { useAppStore } from '../store/appStore';

export const LandingPage: React.FC = () => {
  const [aboutExpanded, setAboutExpanded] = useState(false);
  const { setCurrentPage } = useAppStore();

  const handleExampleClick = () => {
    setCurrentPage(2);
  };

  const handleCustomClick = () => {
    setCurrentPage(3);
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 gradient-border">
      <div className="w-full max-w-2xl animate-fade-in relative z-10">
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-800 mb-6 lowercase">
            hot diggity derm
          </h1>
          
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 mb-8 max-w-xs mx-auto">
            <button
              onClick={() => setAboutExpanded(!aboutExpanded)}
              className="w-full flex items-center justify-center text-center p-2 hover:bg-gray-50 rounded-xl transition-colors duration-200"
            >
              <span className="text-xs font-medium text-gray-700 lowercase">about</span>
              <span className={`transform transition-transform duration-300 text-xs ml-2 ${aboutExpanded ? 'rotate-180' : ''}`}>
                ↓
              </span>
            </button>
            
            <div className={`overflow-hidden transition-all duration-300 ${
              aboutExpanded ? 'max-h-16 opacity-100' : 'max-h-0 opacity-0'
            }`}>
              <div className="px-2 pb-2 pt-1 border-t border-gray-100 text-center">
                <p className="text-soft-grey leading-relaxed text-xs lowercase">
                  ai-powered dermatology analysis using monet for instant skin condition insights.
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md mx-auto">
            <Button
              onClick={handleExampleClick}
              variant="primary"
              size="lg"
              className="w-full"
            >
              view examples
            </Button>
            
            <Button
              onClick={handleCustomClick}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              upload custom
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
