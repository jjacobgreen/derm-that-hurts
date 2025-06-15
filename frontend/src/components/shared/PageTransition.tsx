
import React, { useEffect, useState } from 'react';

interface PageTransitionProps {
  children: React.ReactNode;
  direction: 'forward' | 'back' | 'none';
  isVisible: boolean;
}

export const PageTransition: React.FC<PageTransitionProps> = ({ 
  children, 
  direction, 
  isVisible 
}) => {
  const [shouldRender, setShouldRender] = useState(isVisible);
  const [animationClass, setAnimationClass] = useState('');

  useEffect(() => {
    if (isVisible) {
      setShouldRender(true);
      // Start enter animation
      setTimeout(() => {
        if (direction === 'back') {
          setAnimationClass('page-enter-home-active');
        } else {
          setAnimationClass('page-enter-active');
        }
      }, 10);
    } else {
      // Start exit animation
      if (direction === 'back') {
        setAnimationClass('page-exit-home-active');
      } else {
        setAnimationClass('page-exit-active');
      }
      
      setTimeout(() => {
        setShouldRender(false);
        setAnimationClass('');
      }, 500);
    }
  }, [isVisible, direction]);

  if (!shouldRender) return null;

  const baseClass = direction === 'back' 
    ? (isVisible ? 'page-enter-home' : 'page-exit-home')
    : (isVisible ? 'page-enter' : 'page-exit');

  return (
    <div className={`${baseClass} ${animationClass}`}>
      {children}
    </div>
  );
};
