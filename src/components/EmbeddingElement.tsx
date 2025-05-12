import React, { useMemo, useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';

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
  // Create a ref for the element to position the slider relative to it
  const elementRef = useRef<HTMLDivElement>(null);
  // State to track position for slider
  const [sliderPosition, setSliderPosition] = useState({ left: 0, top: 0 });

  // Update slider position when the element is selected
  useEffect(() => {
    if (isSelected && elementRef.current) {
      const updatePosition = () => {
        const rect = elementRef.current?.getBoundingClientRect();
        if (rect) {
          // Position in the center below the element
          setSliderPosition({
            left: rect.left + rect.width / 2,
            top: rect.bottom + 5 // Add a small gap
          });
        }
      };

      // Initial position
      updatePosition();

      // Keep updating on scroll/resize to ensure slider follows element
      window.addEventListener('scroll', updatePosition, { passive: true });
      window.addEventListener('resize', updatePosition, { passive: true });

      return () => {
        window.removeEventListener('scroll', updatePosition);
        window.removeEventListener('resize', updatePosition);
      };
    }
  }, [isSelected]);
  // Calculate the color based on the value
  const { backgroundColor, textColor } = useMemo(() => {
    // Clamp the value to the maximum range for color interpolation
    const clampedValue = Math.max(-maxAbsValue, Math.min(maxAbsValue, value));

    // Normalize to [-1, 1] range
    const normalizedValue = clampedValue / maxAbsValue;

    // Use a continuous color gradient from red (negative) to gray (zero) to blue (positive)
    if (normalizedValue < 0) {
      // Negative values: pink/red, using exponential intensity to emphasize values closer to zero
      // Apply a power transformation to dramatically increase intensity for smaller values
      const intensity = Math.pow(Math.min(1, -normalizedValue), 0.35) * 1.2;
      return {
        backgroundColor: `rgb(${Math.round(240 + 15 * intensity)}, ${Math.round(240 - 100 * intensity)}, ${Math.round(240 - 100 * intensity)})`,
        textColor: 'black' // Always black text
      };
    } else if (normalizedValue > 0) {
      // Positive values: blue, using exponential intensity to emphasize values closer to zero
      // Apply a power transformation to dramatically increase intensity for smaller values
      const intensity = Math.pow(Math.min(1, normalizedValue), 0.35) * 1.2;
      return {
        backgroundColor: `rgb(${Math.round(240 - 100 * intensity)}, ${Math.round(240 - 100 * intensity)}, ${Math.round(240 + 15 * intensity)})`,
        textColor: 'black' // Always black text
      };
    } else {
      // Exactly zero: light gray
      return {
        backgroundColor: 'rgb(240, 240, 240)',
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
  // Use consistent size regardless of the size prop - zoomed out to about 60%
  const sizeStyles = useMemo(() => {
    const baseClasses = 'rounded-md font-mono flex flex-col justify-center items-center';

    // Use a single, much smaller size for all elements
    return {
      container: `${baseClasses} py-0.5 px-0.5`,
      coefficient: 'text-[0.45rem]',  // Ultra small text
      exponent: 'text-[0.45rem] mt-0',  // Same size, no margin between parts
      minWidth: '1.9rem',
      height: '1.7rem',
      width: '1.9rem'  // Smaller fixed width for all elements (approximately 60% of original)
    };
  }, []);

  return (
    <div className="relative">
      <div
        ref={elementRef}
        className={`${sizeStyles.container} ${selectable ? 'cursor-pointer' : ''} ${isSelected ? 'border-2 border-fuchsia-500' : 'border-2 border-transparent'}`}
        style={{
          backgroundColor,
          color: textColor,
          width: sizeStyles.width,
          height: sizeStyles.height,
          boxShadow: isSelected ? '0 0 4px rgba(217, 70, 239, 0.6)' : '0 1px 1px rgba(0, 0, 0, 0.05)'
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
      {isSelected && onValueChange && createPortal(
        <div className="fixed bg-white rounded-md shadow-lg p-1.5 w-20 animate-fadeIn z-50"
             style={{
               left: `${sliderPosition.left}px`,
               top: `${sliderPosition.top}px`,
               transform: 'translateX(-50%)'
             }}>
          {valueLabel && (
            <div className="text-[0.6rem] font-semibold text-gray-700 mb-0.5 text-center">
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
            className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
          />
          <div className="flex justify-between text-[0.55rem] text-gray-500 mt-0.5">
            <span>{-maxAbsValue.toFixed(1)}</span>
            <span>{maxAbsValue.toFixed(1)}</span>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
};

export default EmbeddingElement;