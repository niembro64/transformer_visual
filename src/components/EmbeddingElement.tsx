import React, { useMemo, useState } from 'react';

interface EmbeddingElementProps {
  value: number;
  /**
   * Optional property to set the maximum absolute value for color scaling
   * Default is 0.5 for neural network weights
   */
  maxAbsValue?: number;
  /**
   * Optional property to set size
   */
  size?: 'sm' | 'md' | 'lg';
  /**
   * Number of significant digits to show in scientific notation
   */
  precision?: number;
  /**
   * Whether this element is selectable
   */
  selectable?: boolean;
  /**
   * Whether this element is currently selected
   */
  isSelected?: boolean;
  /**
   * Callback when element is clicked (for selection)
   */
  onClick?: () => void;
  /**
   * Callback when the value changes via slider
   */
  onValueChange?: (newValue: number) => void;
  /**
   * Optional label for the element when selected (e.g., "Token.Dimension")
   */
  valueLabel?: string;
}

/**
 * Component to visualize a single value in an embedding vector
 * Uses color interpolation to represent the magnitude and sign:
 * - Extremely light pink for negative values
 * - Extremely light gray for values near zero
 * - Extremely light blue for positive values
 *
 * Visual indicator for selected state:
 * - Thick magenta border only appears around the currently selected element
 * - All other elements have transparent borders
 * - Cursor changes to pointer when hovering over selectable elements
 *
 * Text is displayed in black for good readability on these light backgrounds
 */
const EmbeddingElement: React.FC<EmbeddingElementProps> = ({
  value,
  maxAbsValue = 0.5,
  size = 'md', // Size prop is maintained for compatibility but not used
  precision = 2,
  selectable = false,
  isSelected = false,
  onClick,
  onValueChange,
  valueLabel
}) => {
  // Calculate the color based on the value
  const { backgroundColor, textColor } = useMemo(() => {
    // Clamp the value to the maximum range for color interpolation
    const clampedValue = Math.max(-maxAbsValue, Math.min(maxAbsValue, value));

    // Normalize to [-1, 1] range
    const normalizedValue = clampedValue / maxAbsValue;

    // Neutral zone threshold - values close to zero will be gray
    const neutralThreshold = 0.1;

    if (normalizedValue < -neutralThreshold) {
      // Negative values: extremely light red
      const intensity = Math.min(1, -normalizedValue * 0.3); // Extremely light intensity
      return {
        backgroundColor: `rgb(${Math.round(240 + 15 * intensity)}, ${Math.round(220 * (1 - intensity))}, ${Math.round(220 * (1 - intensity))})`,
        textColor: 'black'
      };
    } else if (normalizedValue > neutralThreshold) {
      // Positive values: extremely light blue
      const intensity = Math.min(1, normalizedValue * 0.3); // Extremely light intensity
      return {
        backgroundColor: `rgb(${Math.round(220 * (1 - intensity))}, ${Math.round(220 * (1 - intensity))}, ${Math.round(240 + 15 * intensity)})`,
        textColor: 'black'
      };
    } else {
      // Values close to zero: extremely light gray
      return {
        backgroundColor: '#e0e0e0',
        textColor: 'black'
      };
    }
  }, [value, maxAbsValue]);

  // Format the value in scientific notation and split into coefficient and exponent
  const { coefficient, exponent } = useMemo(() => {
    // Check if the value is zero (special case)
    if (value === 0) {
      return { coefficient: '0.00', exponent: 'e+0' };
    }

    // Format to scientific notation
    const scientificNotation = value.toExponential(precision);

    // Split at 'e' to separate coefficient from exponent
    const [coef, exp] = scientificNotation.split('e');

    return {
      coefficient: coef,
      exponent: `e${exp}`
    };
  }, [value, precision]);

  // Determine styling based on size
  // Use consistent size regardless of the size prop
  const sizeStyles = useMemo(() => {
    const baseClasses = 'rounded-lg font-mono flex flex-col justify-center items-center';

    // Use a single, smaller size for all elements
    return {
      container: `${baseClasses} py-1 px-1`,
      coefficient: 'text-[0.65rem]',  // Extra small text
      exponent: 'text-[0.65rem] mt-0',  // Same size, no margin between parts
      minWidth: '3.2rem',
      height: '2.8rem',
      width: '3.2rem'  // Smaller fixed width for all elements
    };
  }, []);

  return (
    <div className="relative">
      <div
        className={`${sizeStyles.container} ${selectable ? 'cursor-pointer' : ''} ${isSelected ? 'border-4 border-fuchsia-500' : 'border-4 border-transparent'}`}
        style={{
          backgroundColor,
          color: textColor,
          width: sizeStyles.width,
          height: sizeStyles.height,
          boxShadow: isSelected ? '0 0 6px rgba(217, 70, 239, 0.6)' : '0 1px 1px rgba(0, 0, 0, 0.05)'
        }}
        onClick={selectable ? onClick : undefined}
      >
        <div className={`${sizeStyles.coefficient} text-center w-full`}>
          {coefficient}
        </div>
        <div className={`${sizeStyles.exponent} text-center w-full`}>
          {exponent}
        </div>
      </div>

      {/* Floating slider overlay that appears below the element when selected */}
      {isSelected && onValueChange && (
        <div className="absolute left-1/2 transform -translate-x-1/2 mt-1 z-10 bg-white rounded-md shadow-lg p-2 w-24 animate-fadeIn"
             style={{ top: '100%' }}>
          {valueLabel && (
            <div className="text-xs font-semibold text-gray-700 mb-1 text-center">
              {valueLabel}
            </div>
          )}
          <input
            type="range"
            min={-maxAbsValue}
            max={maxAbsValue}
            step={0.01}
            value={value}
            onChange={(e) => onValueChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{-maxAbsValue.toFixed(1)}</span>
            <span>{maxAbsValue.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmbeddingElement;