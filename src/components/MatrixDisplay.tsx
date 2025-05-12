import React from 'react';
import EmbeddingElement from './EmbeddingElement';

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
  cellSize?: 'sm' | 'md' | 'lg';
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
}) => {
  if (!data || data.length === 0) {
    return <div>No data to display</div>;
  }

  const rows = data.length;
  const cols = data[0].length;

  // Determine if we need to render row/column labels
  const showRowLabels = rowLabels && rowLabels.length === rows;
  const showColumnLabels = columnLabels && columnLabels.length === cols;

  // Use much smaller cell sizes with minimal spacing (about 60% of original size)
  const cellWidth = 1.9; // Matches element width exactly
  const cellHeight = 1.7; // Matches element height exactly
  const cellGap = 0.03; // Extremely minimal gap between cells

  // Calculate grid template columns for the entire grid including row labels
  const gridTemplateColumns = showRowLabels
    ? `1.5rem repeat(${cols}, ${cellWidth}rem)`  // First column for row labels (narrower)
    : `repeat(${cols}, ${cellWidth}rem)`;

  // Calculate grid template rows including header for column labels
  const gridTemplateRows = showColumnLabels
    ? `0.7rem repeat(${rows}, ${cellHeight}rem)` // First row for column labels (shorter)
    : `repeat(${rows}, ${cellHeight}rem)`;

  return (
    <div className={`flex flex-col ${className} overflow-hidden`}>
      {/* Matrix Label */}
      {label && (
        <div className="text-center text-[0.6rem] font-semibold text-gray-700 mb-1">{label}</div>
      )}

      {/* Main grid container */}
      <div
        className="grid"
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
              className="text-center text-[0.4rem] text-gray-500 flex items-center justify-center"
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
                className="text-center text-[0.4rem] text-gray-500 flex items-center justify-center"
                style={{
                  gridColumn: 1,
                  gridRow: showColumnLabels ? i + 2 : i + 1
                }}
              >
                {rowLabels![i]}
              </div>
            )}

            {/* Matrix cells for this row */}
            {row.map((value, j) => (
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
                  isSelected={selectedElement !== null &&
                              selectedElement.matrixType === matrixType &&
                              selectedElement.row === i &&
                              selectedElement.col === j}
                  onClick={() => onElementClick && matrixType && onElementClick(matrixType, i, j)}
                  onValueChange={selectedElement !== null &&
                                 selectedElement.matrixType === matrixType &&
                                 selectedElement.row === i &&
                                 selectedElement.col === j &&
                                 onValueChange ?
                                 onValueChange : undefined}
                  valueLabel={undefined}
                  /* We provide valueLabel from the parent component now */
                />
              </div>
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default MatrixDisplay;