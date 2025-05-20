import React from 'react';
import MatrixDisplay from './MatrixDisplay';

interface TokenComponentProps {
  text: string;
  embedding: number[];
  embeddingDim: number;
  isSelected?: boolean;
  isHighlighted?: boolean;
  onClick?: () => void;
  variant?: 'default' | 'input' | 'prediction';
}

const TokenComponent: React.FC<TokenComponentProps> = ({
  text,
  embedding,
  embeddingDim,
  isSelected = false,
  isHighlighted = false,
  onClick,
  variant = 'default',
}) => {
  // Determine style based on variant
  const getBorderStyle = () => {
    if (isHighlighted) return 'border-blue-500';
    
    switch (variant) {
      case 'input':
        return 'border-gray-400 hover:border-red-300';
      case 'prediction':
        return 'border-gray-300';
      default:
        return 'border-gray-300';
    }
  };

  const getBackgroundStyle = () => {
    switch (variant) {
      case 'input':
        return 'bg-white hover:bg-red-50';
      case 'prediction':
        return 'bg-gray-100 hover:bg-gray-200';
      default:
        return 'bg-gray-100 hover:bg-gray-200';
    }
  };

  return (
    <div
      className={`
        px-2 sm:px-3 py-1 sm:py-1.5 
        border ${getBorderStyle()} 
        rounded text-xs sm:text-sm 
        min-w-[2.5rem] sm:min-w-[3.5rem] 
        text-center shadow-sm 
        ${getBackgroundStyle()} 
        font-mono cursor-pointer 
        transition-all group relative
      `}
      onClick={onClick}
    >
      {text}
      {/* Show embedding as matrix on hover */}
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
        <div className="mb-1 text-gray-600 text-center font-medium">
          {text} embedding
        </div>
        <MatrixDisplay
          data={[embedding]}
          rowLabels={['']}
          columnLabels={Array.from(
            { length: embeddingDim },
            (_, i) => `d${i + 1}`
          )}
          maxAbsValue={0.2}
          cellSize="xs"
          selectable={false}
          matrixType="none"
        />
      </div>
    </div>
  );
};

export default TokenComponent;