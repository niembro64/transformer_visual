import React, {
  useMemo,
  useState,
  useEffect,
  useRef,
  useCallback,
} from 'react';
import { createPortal } from 'react-dom';
import { isPortraitOrientation } from '../utils/matrixOperations';

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
  size?: 'xs' | 'sm' | 'md' | 'lg';
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
  /**
   * Whether to start oscillating automatically when selected
   */
  autoOscillate?: boolean;
  /**
   * Whether we are in training mode (disables wiggle when true)
   */
  isTrainingMode?: boolean;
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
  valueLabel,
  autoOscillate = false,
  isTrainingMode = false,
}) => {
  // Create a ref for the element to position the slider relative to it
  const elementRef = useRef<HTMLDivElement>(null);
  // State to track position for slider
  const [sliderPosition, setSliderPosition] = useState({ left: 0, top: 0 });
  // State to track oscillation
  const [isOscillating, setIsOscillating] = useState(false);
  // Reference to animation frame ID for cleanup
  const animationFrameRef = useRef<number | null>(null);
  // Reference to track the animation start time
  const startTimeRef = useRef<number>(0);

  // Start oscillation animation - using a smooth sine wave
  const startOscillation = useCallback(() => {
    if (!onValueChange) return;

    setIsOscillating(true);
    startTimeRef.current = performance.now();

    const animate = (time: number) => {
      // Calculate sine wave value (oscillating between -10 and 10)
      const elapsedTime = time - startTimeRef.current;
      // Complete one cycle every 3 seconds (3000ms)
      const frequency = 1000; // Slower frequency for smoother oscillation
      const progress = (elapsedTime % frequency) / frequency;

      // Oscillate between -10 and 10 regardless of maxAbsValue
      const TARGET_MAX_VALUE = 10;
      const oscillatedValue =
        TARGET_MAX_VALUE * Math.sin(progress * Math.PI * 2);

      // This forces the value to be exactly what we want, and other components
      // like the random walk won't interfere with it
      requestAnimationFrame(() => {
        // Update the value inside another animation frame to ensure it takes priority
        onValueChange(oscillatedValue);
      });

      // Schedule next frame
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    // Start animation
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [onValueChange]);

  // Stop oscillation
  const stopOscillation = useCallback(() => {
    setIsOscillating(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, []);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Track if oscillation has happened, to avoid repeating on re-renders
  const hasAutoOscillated = useRef(false);

  // Stop oscillation when entering training mode
  useEffect(() => {
    if (isTrainingMode && isOscillating) {
      stopOscillation();
    }
  }, [isTrainingMode, isOscillating, stopOscillation]);

  // Update slider position when the element is selected or orientation changes
  useEffect(() => {
    if (isSelected && elementRef.current && !isTrainingMode) {
      const updatePosition = () => {
        const rect = elementRef.current?.getBoundingClientRect();
        if (rect) {
          const isPortrait = isPortraitOrientation();
          // Position based on screen orientation - below on landscape, to the right on portrait
          if (isPortrait) {
            setSliderPosition({
              left: rect.right + 10, // Place to the right with a small gap in portrait mode
              top: rect.top + rect.height / 2,
            });
          } else {
            setSliderPosition({
              left: rect.left + rect.width / 2,
              top: rect.bottom + 5, // Below with a small gap in landscape mode
            });
          }
        }
      };

      // Initial position
      updatePosition();

      // Auto start oscillation ONLY on initial page load for the designated element
      if (autoOscillate && onValueChange && !hasAutoOscillated.current) {
        hasAutoOscillated.current = true;
        setTimeout(() => startOscillation(), 100); // Short delay to ensure everything is rendered
      }

      // Keep updating on scroll/resize to ensure slider follows element
      window.addEventListener('scroll', updatePosition, { passive: true });
      window.addEventListener('resize', updatePosition, { passive: true });

      // Check orientation changes
      const checkOrientation = () => {
        // Simply update position on orientation change
        updatePosition();
      };

      window.addEventListener('orientationchange', checkOrientation, {
        passive: true,
      });

      return () => {
        window.removeEventListener('scroll', updatePosition);
        window.removeEventListener('resize', updatePosition);
        window.removeEventListener('orientationchange', checkOrientation);
      };
    } else {
      // When element is deselected, stop oscillation
      stopOscillation();
    }
  }, [
    isSelected,
    stopOscillation,
    autoOscillate,
    onValueChange,
    startOscillation,
    isTrainingMode,
  ]);

  // Calculate the color based on a sinusoidal mapping function that provides
  // stronger visual differentiation near zero and asymptotically approaches
  // pure blue/red at the extremes
  const { backgroundColor, textColor } = useMemo(() => {
    // Check if we're dealing with a positional encoding or softmax value that should use a different scale
    // We'll determine this based on the maxAbsValue - if it's 1.0, we're dealing with positional/softmax
    const isPosEncodingOrSoftmax = maxAbsValue === 1.0;

    // Use arctangent function to create a smooth, non-linear mapping
    // that approaches asymptotic limits but changes more dramatically near zero

    // Apply sigmoid-like transformation using atan function
    // atan maps the entire real line to [-π/2, π/2], which we'll normalize to [-1, 1]
    // We'll adjust the steepness of the curve with a scaling factor
    const steepness = isPosEncodingOrSoftmax ? 2.0 : 0.3; // Higher steepness for pos encoding/softmax

    // Get value between -1 and 1 using atan function
    // This creates a sinusoidal-like curve that changes faster near 0
    const normalizedValue = Math.atan(value * steepness) / (Math.PI / 2);

    // Base colors - neutral for zero, vibrant for extremes
    const neutralColor = [240, 240, 240]; 
    const maxBlueColor = [20, 20, 255];
    const maxRedColor = [255, 20, 20];
    // const maxBlueColor = isPosEncodingOrSoftmax ? [0, 0, 0] : [20, 20, 255];
    // const maxRedColor = isPosEncodingOrSoftmax ? [255, 20, 20] : [255, 20, 20];
    // Compute the color based on normalized value
    let red, green, blue;

    if (normalizedValue < 0) {
      // Negative values - interpolate between neutral and max red
      const t = -normalizedValue; // How far toward max red (0 to 1)
      red = neutralColor[0] * (1 - t) + maxRedColor[0] * t;
      green = neutralColor[1] * (1 - t) + maxRedColor[1] * t;
      blue = neutralColor[2] * (1 - t) + maxRedColor[2] * t;

      // For very negative values, make text white for better contrast
      const textColorThreshold = isPosEncodingOrSoftmax ? -0.5 : -0.7; // Lower threshold for pos encodings
      return {
        backgroundColor: `rgb(${Math.round(red)}, ${Math.round(
          green
        )}, ${Math.round(blue)})`,
        textColor: normalizedValue < textColorThreshold ? 'white' : 'black',
      };
    } else if (normalizedValue > 0) {
      // Positive values - interpolate between neutral and max blue
      const t = normalizedValue; // How far toward max blue (0 to 1)
      red = neutralColor[0] * (1 - t) + maxBlueColor[0] * t;
      green = neutralColor[1] * (1 - t) + maxBlueColor[1] * t;
      blue = neutralColor[2] * (1 - t) + maxBlueColor[2] * t;

      // For very positive values, make text white for better contrast
      const textColorThreshold = isPosEncodingOrSoftmax ? 0.5 : 0.7; // Lower threshold for pos encodings
      return {
        backgroundColor: `rgb(${Math.round(red)}, ${Math.round(
          green
        )}, ${Math.round(blue)})`,
        textColor: normalizedValue > textColorThreshold ? 'white' : 'black',
      };
    } else {
      // Exactly zero: neutral gray
      return {
        backgroundColor: `rgb(${neutralColor[0]}, ${neutralColor[1]}, ${neutralColor[2]})`,
        textColor: 'black',
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

    // Add "+" prefix for positive values, "-" is already included for negative values
    let formattedCoef = value > 0 ? `+${coef}` : coef;

    // Ensure coefficient is always 5 characters for consistent sizing/centering
    // Format should be [sign][digit].[digits] where we have 2 digits after decimal point
    if (formattedCoef.length < 5) {
      // If shorter than 5 chars, pad with zeros after decimal point
      const parts = formattedCoef.split('.');
      if (parts.length === 2) {
        while (formattedCoef.length < 5) {
          parts[1] += '0';
          formattedCoef = parts[0] + '.' + parts[1];
        }
      }
    } else if (formattedCoef.length > 5) {
      // If longer than 5 chars, truncate decimal places
      const sign = formattedCoef[0];
      const firstDigit = formattedCoef[1];
      formattedCoef = sign + firstDigit + '.' + formattedCoef.slice(3, 5);
    }

    return {
      coefficient: formattedCoef,
      exponent: `e${exp}`,
    };
  }, [value, precision]);

  // Determine styling based on size and device
  // Use size-appropriate styling with mobile responsiveness
  const sizeStyles = useMemo(() => {
    const baseClasses =
      'rounded-md font-mono flex flex-col justify-between items-center';

    // Check if we're on a mobile device or in portrait orientation
    const isPortrait = isPortraitOrientation();

    // Size configurations with mobile variants - more compact in portrait mode
    const sizeMap = {
      xs: {
        container: `${baseClasses} py-0.5 px-0.5`,
        coefficient: isPortrait ? 'text-[0.42rem]' : 'text-[0.45rem]',
        exponent: isPortrait ? 'text-[0.37rem]' : 'text-[0.45rem]',
        minWidth: isPortrait ? '1.5rem' : '1.7rem',
        height: isPortrait ? '1.4rem' : '1.5rem',
        width: isPortrait ? '1.5rem' : '1.7rem',
      },
      sm: {
        container: `${baseClasses} py-0.5 px-0.5`,
        coefficient: isPortrait ? 'text-[0.45rem]' : 'text-[0.5rem]',
        exponent: isPortrait ? 'text-[0.4rem]' : 'text-[0.5rem]',
        minWidth: isPortrait ? '1.7rem' : '2.1rem',
        height: isPortrait ? '1.6rem' : '1.9rem',
        width: isPortrait ? '1.7rem' : '2.1rem',
      },
      md: {
        container: `${baseClasses} py-0.5 px-0.5`,
        coefficient: isPortrait ? 'text-[0.48rem]' : 'text-[0.55rem]',
        exponent: isPortrait ? 'text-[0.43rem]' : 'text-[0.55rem]',
        minWidth: isPortrait ? '1.9rem' : '2.7rem',
        height: isPortrait ? '1.7rem' : '2.4rem',
        width: isPortrait ? '1.9rem' : '2.7rem',
      },
      lg: {
        container: `${baseClasses} py-0.5 px-0.5`,
        coefficient: isPortrait ? 'text-[0.52rem]' : 'text-[0.6rem]',
        exponent: isPortrait ? 'text-[0.47rem]' : 'text-[0.6rem]',
        minWidth: isPortrait ? '2.1rem' : '3.4rem',
        height: isPortrait ? '1.9rem' : '3.0rem',
        width: isPortrait ? '2.1rem' : '3.4rem',
      },
    };

    return sizeMap[size];
  }, [size]);

  return (
    <div className="relative">
      <div
        ref={elementRef}
        className={`${sizeStyles.container} ${
          selectable && !isTrainingMode ? 'cursor-pointer' : ''
        } ${
          isSelected
            ? 'border-2 border-fuchsia-500'
            : 'border-[1px] border-white'
        }`}
        style={{
          backgroundColor,
          color: textColor,
          width: sizeStyles.width,
          height: sizeStyles.height,
          // boxShadow: isSelected
          //   ? '0 0 4px rgba(217, 70, 239, 0.6)'
          //   : '0 1px 1px rgba(0, 0, 0, 0.05)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          padding: '0.125rem',
        }}
        onClick={selectable && !isTrainingMode ? onClick : undefined}
      >
        <div
          className={`${sizeStyles.coefficient} flex justify-center items-center w-full font-mono tracking-tight`}
        >
          <span className="block text-center leading-none">{coefficient}</span>
        </div>
        <div
          className={`${sizeStyles.exponent} flex justify-center items-center w-full font-mono tracking-tight`}
        >
          <span className="block text-center leading-none">{exponent}</span>
        </div>
      </div>

      {/* Floating slider overlay that appears below or to the side of the element when selected */}
      {isSelected &&
        onValueChange &&
        !isTrainingMode &&
        createPortal(
          <div
            className="fixed bg-white rounded-md shadow-lg p-1.5 animate-fadeIn z-50"
            style={{
              left: `${sliderPosition.left}px`,
              top: `${sliderPosition.top}px`,
              transform: isPortraitOrientation()
                ? 'translateY(-50%)' // Center vertically in portrait mode
                : 'translateX(-50%)', // Center horizontally in landscape mode
              width: isPortraitOrientation() ? '6rem' : '8rem',
            }}
          >
            {valueLabel && (
              <div className="text-[0.6rem] font-semibold text-gray-700 mb-0.5 text-center">
                {valueLabel}
              </div>
            )}

            <input
              title='"Drag to adjust value"'
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={value}
              onChange={(e) => onValueChange(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[0.55rem] text-gray-500 mt-0.5">
              <span>-10.0</span>
              <span>10.0</span>
            </div>

            {!isTrainingMode && (
              <button
                onClick={() =>
                  isOscillating ? stopOscillation() : startOscillation()
                }
                className={`w-full mt-1.5 py-0.5 px-1 text-[0.55rem] font-medium rounded-sm ${
                  isOscillating
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                } transition-colors`}
                title="Oscillate this value using sine waves (only available in inference mode)"
              >
                {isOscillating ? 'Stop' : 'Wiggle'}
              </button>
            )}
          </div>,
          document.body
        )}
    </div>
  );
};

export default EmbeddingElement;
