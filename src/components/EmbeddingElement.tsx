import React, { useMemo } from 'react';

interface EmbeddingElementProps {
  value: number;
  /**
   * Optional property to set the maximum absolute value for color scaling
   * Default is 3 (values beyond -3 or +3 will use the most intense colors)
   */
  maxAbsValue?: number;
  /**
   * Optional property to set size
   */
  size?: 'sm' | 'md' | 'lg';
}

/**
 * Component to visualize a single value in an embedding vector
 * Uses color interpolation to represent the magnitude and sign:
 * - Red for negative values
 * - White for values near zero
 * - Blue for positive values
 */
const EmbeddingElement: React.FC<EmbeddingElementProps> = ({
  value,
  maxAbsValue = 3,
  size = 'md'
}) => {
  // Calculate the color based on the value
  const backgroundColor = useMemo(() => {
    // Clamp the value to the maximum range for color interpolation
    const clampedValue = Math.max(-maxAbsValue, Math.min(maxAbsValue, value));
    
    // Normalize to [-1, 1] range
    const normalizedValue = clampedValue / maxAbsValue;
    
    if (normalizedValue < 0) {
      // Negative values: interpolate from white to red
      const intensity = Math.round(-normalizedValue * 255);
      return `rgb(255, ${255 - intensity}, ${255 - intensity})`;
    } else {
      // Positive values: interpolate from white to blue
      const intensity = Math.round(normalizedValue * 255);
      return `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
    }
  }, [value, maxAbsValue]);

  // Determine styling based on size
  const sizeClasses = {
    sm: 'p-1 text-xs',
    md: 'p-2 text-sm',
    lg: 'p-3 text-base'
  }[size];

  return (
    <div 
      className={`rounded-lg ${sizeClasses} flex items-center justify-center font-mono`}
      style={{ 
        backgroundColor,
        minWidth: size === 'sm' ? '2rem' : size === 'md' ? '2.5rem' : '3rem',
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
      }}
    >
      {value.toFixed(2)}
    </div>
  );
};

export default EmbeddingElement;