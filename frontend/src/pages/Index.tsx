
import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { LandingPage } from './LandingPage';
import { ExampleSelection } from './ExampleSelection';
import { CustomUpload } from './CustomUpload';
import { ImagePreview } from './ImagePreview';
import { LoadingPage } from './LoadingPage';
import { ResultsReport } from './ResultsReport';

const Index = () => {
  const { currentPage } = useAppStore();
  const [previousPage, setPreviousPage] = useState(1);
  const [transitionDirection, setTransitionDirection] = useState<'forward' | 'back' | 'none'>('none');

  useEffect(() => {
    if (currentPage !== previousPage) {
      if (currentPage === 1) {
        setTransitionDirection('back');
      } else if (currentPage > previousPage) {
        setTransitionDirection('forward');
      } else {
        setTransitionDirection('back');
      }
      setPreviousPage(currentPage);
    }
  }, [currentPage, previousPage]);

  const renderPage = () => {
    switch (currentPage) {
      case 1:
        return <LandingPage />;
      case 2:
        return <ExampleSelection />;
      case 3:
        return <CustomUpload />;
      case 4:
        return <ImagePreview />;
      case 5:
        return <LoadingPage />;
      case 6:
        return <ResultsReport />;
      default:
        return <LandingPage />;
    }
  };

  return (
    <div className="min-h-screen bg-off-white overflow-hidden">
      <div className={`transition-transform duration-500 ease-out ${
        transitionDirection === 'forward' ? 'animate-[slideUp_0.5s_ease-out]' :
        transitionDirection === 'back' ? 'animate-[slideDown_0.5s_ease-out]' : ''
      }`}>
        {renderPage()}
      </div>
    </div>
  );
};

export default Index;
