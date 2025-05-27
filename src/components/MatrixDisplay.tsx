import React from 'react';
import EmbeddingElement from './EmbeddingElement';
import { isPortraitOrientation } from '../utils/matrixOperations';

interface MatrixDisplayProps {
  data: number[][];
  label?: string;
  /**
   * Optional property to show row labels
   */
  rowLabels?: string[];
  /**
   * Optional property to show column labels
   */
  columnLabels?: string[];
  /**
   * Max absolute value for color scaling in the EmbeddingElements
   */
  maxAbsValue?: number;
  /**
   * Size of each cell
   */
  cellSize?: 'xs' | 'sm' | 'md' | 'lg';
  /**
   * CSS class for additional styling
   */
  className?: string;
  /**
   * Whether elements in this matrix are selectable
   */
  selectable?: boolean;
  /**
   * Currently selected element coordinates or null if none selected
   */
  selectedElement?: {
    matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'weightW1' | 'weightW2' | 'none';
    row: number;
    col: number;
  } | null;
  /**
   * The matrix type for this matrix display
   */
  matrixType?: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'weightW1' | 'weightW2' | 'none';
  /**
   * Callback when an element is clicked
   */
  onElementClick?: (matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'weightW1' | 'weightW2' | 'none', row: number, col: number) => void;
  /**
   * Callback when element value changes via slider
   */
  onValueChange?: (newValue: number) => void;
  /**
   * Label for the currently selected value
   */
  valueLabel?: string;
  /**
   * Whether to auto-oscillate the selected value
   */
  autoOscillate?: boolean;
  /**
   * Whether we are in training mode (disables wiggle when true)
   */
  isTrainingMode?: boolean;
}

/**
 * Component to display a matrix of values
 * Uses a CSS grid to show the matrix elements with proper alignment of row and column labels
 */
const MatrixDisplay: React.FC<MatrixDisplayProps> = ({
  data,
  label,
  rowLabels,
  columnLabels,
  maxAbsValue = 3,
  cellSize = 'md', // Maintained for compatibility but not used for sizing
  className = '',
  selectable = false,
  selectedElement = null,
  matrixType,
  onElementClick,
  onValueChange,
  valueLabel,
  autoOscillate = false,
  isTrainingMode = false,
}) => {
  if (!data || data.length === 0) {
    return <div>No data to display</div>;
  }

  const rows = data.length;
  const cols = data[0].length;

  // Determine if we need to render row/column labels
  const showRowLabels = rowLabels && rowLabels.length === rows;
  const showColumnLabels = columnLabels && columnLabels.length === cols;

  // Use size-appropriate cell sizes with minimal spacing, responsive to screen size
  const sizeMap = {
    xs: { 
      default: { width: 1.7, height: 1.5 },  // Extra small for hidden layers
      mobile: { width: 1.4, height: 1.3 }    // Slightly smaller on mobile but with better spacing
    },
    sm: { 
      default: { width: 2.1, height: 1.9 },  // Small for most elements
      mobile: { width: 1.6, height: 1.5 }    // Mobile size
    },
    md: { 
      default: { width: 2.7, height: 2.4 },  // Medium if needed
      mobile: { width: 1.8, height: 1.6 }    // Mobile size
    },
    lg: { 
      default: { width: 3.4, height: 3.0 },  // Large if needed
      mobile: { width: 2.0, height: 1.8 }    // Mobile size
    }
  };

  // Check if we're on a mobile device (using window.innerWidth) or in portrait orientation
  const isMobile = typeof window !== 'undefined' && (window.innerWidth < 768 || isPortraitOrientation());
  const sizeVariant = isMobile ? 'mobile' : 'default';

  const cellWidth = sizeMap[cellSize][sizeVariant].width;  // Use width based on size and device
  const cellHeight = sizeMap[cellSize][sizeVariant].height; // Use height based on size and device
  const cellGap = isMobile ? 0.05 : 0.03; // Slightly larger gap on mobile for better spacing

  // Calculate grid template columns for the entire grid including row labels
  const gridTemplateColumns = showRowLabels
    ? `${isMobile ? '2rem' : '1.5rem'} repeat(${cols}, ${cellWidth}rem)`  // Wider column for row labels on mobile
    : `repeat(${cols}, ${cellWidth}rem)`;

  // Calculate grid template rows including header for column labels
  const gridTemplateRows = showColumnLabels
    ? `${isMobile ? '1rem' : '0.7rem'} repeat(${rows}, ${cellHeight}rem)` // Taller header on mobile
    : `repeat(${rows}, ${cellHeight}rem)`;

  return (
    <div className={`flex flex-col items-center justify-center ${className} overflow-hidden w-full h-full`}>
      {/* Matrix Label */}
      {label && (
        <div className={`text-center ${isMobile ? 'text-[0.65rem]' : 'text-[0.6rem]'} font-semibold text-gray-700 mb-1`}>{label}</div>
      )}

      {/* Main grid container */}
      <div
        className="grid mx-auto"
        style={{
          gridTemplateColumns,
          gridTemplateRows,
          gap: `${cellGap}rem`,
          justifyItems: 'center',
          alignItems: 'center',
          fontSize: 'small'
        }}
      >
        {/* Empty cell in top-left corner when both labels are shown */}
        {showRowLabels && showColumnLabels && (
          <div className="col-start-1 row-start-1"></div>
        )}
        
        {/* Column labels row */}
        {showColumnLabels && 
          columnLabels!.map((label, j) => (
            <div
              key={`col-${j}`}
              className={`text-center ${isMobile ? 'text-[0.55rem]' : 'text-[0.5rem]'} text-gray-600 flex items-center justify-center font-medium`}
              style={{
                gridColumn: showRowLabels ? j + 2 : j + 1,
                gridRow: 1
              }}
            >
              {label}
            </div>
          ))
        }
        
        {/* Row labels and matrix cells */}
        {data.map((row, i) => (
          <React.Fragment key={`row-${i}`}>
            {/* Row label */}
            {showRowLabels && (
              <div
                className={`text-center ${isMobile ? 'text-[0.55rem]' : 'text-[0.5rem]'} text-gray-600 flex items-center justify-center font-medium`}
                style={{
                  gridColumn: 1,
                  gridRow: showColumnLabels ? i + 2 : i + 1
                }}
              >
                {rowLabels![i]}
              </div>
            )}

            {/* Matrix cells for this row */}
            {row.map((value, j) => {
              // Determine if this specific cell is selected
              const isSelected = selectedElement !== null &&
                              selectedElement.matrixType === matrixType &&
                              selectedElement.row === i &&
                              selectedElement.col === j;
              
              // Only provide onValueChange if this cell is selected
              const cellValueChangeHandler = isSelected && onValueChange ? onValueChange : undefined;
              
              return (
                <div
                  key={`cell-${i}-${j}`}
                  className="flex items-center justify-center"
                  style={{
                    gridColumn: showRowLabels ? j + 2 : j + 1,
                    gridRow: showColumnLabels ? i + 2 : i + 1
                  }}
                >
                  <EmbeddingElement
                    value={value}
                    maxAbsValue={maxAbsValue}
                    size={cellSize}
                    selectable={selectable}
                    isSelected={isSelected}
                    onClick={() => onElementClick && matrixType && onElementClick(matrixType, i, j)}
                    onValueChange={cellValueChangeHandler}
                    valueLabel={isSelected ? valueLabel : undefined}
                    autoOscillate={isSelected && autoOscillate}
                    isTrainingMode={isTrainingMode}
                  />
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default MatrixDisplay;