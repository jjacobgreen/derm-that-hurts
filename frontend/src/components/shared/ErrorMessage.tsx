
import React from 'react';
import { Button } from './Button';

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
  onBack?: () => void;
}

export const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  onRetry,
  onBack
}) => {
  return (
    <div className="text-center py-12">
      <div className="derm-card max-w-md mx-auto">
        <div className="text-red-500 text-lg mb-4">⚠️ Something went wrong</div>
        <p className="text-soft-grey mb-6">{message}</p>
        <div className="space-y-3">
          {onRetry && (
            <Button onClick={onRetry} className="w-full">
              Try Again
            </Button>
          )}
          {onBack && (
            <Button variant="cancel" onClick={onBack} className="w-full">
              Go Back
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
