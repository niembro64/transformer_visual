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
}

/**
 * Component to display a matrix of values
 * Uses a CSS grid to show the matrix elements with proper alignment
 */
const MatrixDisplay: React.FC<MatrixDisplayProps> = ({
  data,
  label,
  rowLabels,
  columnLabels,
  maxAbsValue = 3,
  cellSize = 'md',
  className = '',
}) => {
  if (!data || data.length === 0) {
    return <div>No data to display</div>;
  }

  const rows = data.length;
  const cols = data[0].length;

  // Determine if we need to render row/column labels
  const showRowLabels = rowLabels && rowLabels.length === rows;
  const showColumnLabels = columnLabels && columnLabels.length === cols;

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Matrix Label */}
      {label && (
        <div className="text-center font-semibold text-gray-700 mb-2">{label}</div>
      )}

      <div className="flex">
        {/* Row Labels Column */}
        {showRowLabels && (
          <div className="flex flex-col justify-around mr-2 text-right">
            {rowLabels!.map((label, index) => (
              <div key={`row-${index}`} className="py-1 text-xs text-gray-500">
                {label}
              </div>
            ))}
          </div>
        )}

        <div className="flex flex-col">
          {/* Column Labels Row */}
          {showColumnLabels && (
            <div 
              className="flex mb-1" 
              style={{ 
                marginLeft: showRowLabels ? '0' : '0', 
                gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` 
              }}
            >
              {columnLabels!.map((label, index) => (
                <div 
                  key={`col-${index}`} 
                  className="text-center text-xs text-gray-500"
                  style={{ 
                    width: cellSize === 'sm' ? '2rem' : cellSize === 'md' ? '2.5rem' : '3rem',
                    marginRight: '0.25rem'
                  }}
                >
                  {label}
                </div>
              ))}
            </div>
          )}

          {/* Matrix Grid */}
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
          >
            {data.flatMap((row, i) =>
              row.map((value, j) => (
                <div key={`${i}-${j}`}>
                  <EmbeddingElement 
                    value={value} 
                    maxAbsValue={maxAbsValue} 
                    size={cellSize} 
                  />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MatrixDisplay;