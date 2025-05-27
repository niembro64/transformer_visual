import React from 'react';
import MatrixDisplay from './MatrixDisplay';

interface TokenProps {
  text: string;
  onClick?: () => void;
  tokenType:
    | 'tokenizer'
    | 'input'
    | 'output_prediction'
    | 'output_target'
    | 'placeholder';
  // For conditional styling in tokenizer
  isTargetToken?: boolean;
  isPredictedToken?: boolean;
  isRecentlyAdded?: boolean;
  // For training mode
  isTrainingMode?: boolean;
  // For embedding tooltip
  showEmbedding?: boolean;
  embedding?: number[];
  embeddingDimension?: number;
  // For input tokens
  includeHeight?: boolean;
}

const Token: React.FC<TokenProps> = ({
  text,
  onClick,
  tokenType,
  isTargetToken = false,
  isPredictedToken = false,
  isRecentlyAdded = false,
  isTrainingMode = false,
  showEmbedding = false,
  embedding = [],
  embeddingDimension = 0,
  includeHeight = false,
}) => {
  const getTokenStyles = () => {
    const baseStyles =
      'px-2 sm:px-3 py-1 sm:py-1.5 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm font-mono cursor-pointer transition-colors group relative';

    // Height for input tokens
    const heightStyles = includeHeight ? 'h-7 sm:h-9' : '';

    switch (tokenType) {
      case 'tokenizer':
        // Special styling for tokenizer based on state
        if (isTrainingMode && isTargetToken) {
          return `${baseStyles} ${heightStyles} border-2 border-green-500 bg-green-100 text-green-900 font-semibold hover:bg-green-200`;
        } else if (isTargetToken) {
          return `${baseStyles} ${heightStyles} border-2 border-green-500 bg-green-100 text-green-900 font-semibold hover:bg-green-200`;
        } else if (isPredictedToken) {
          return `${baseStyles} ${heightStyles} border-2 border-blue-500 bg-blue-100 text-blue-900 font-semibold hover:bg-blue-200`;
        } else if (isRecentlyAdded) {
          return `${baseStyles} ${heightStyles} border border-blue-500 bg-gray-100 text-gray-800 hover:bg-gray-200`;
        } else {
          return `${baseStyles} ${heightStyles} border border-gray-300 bg-gray-100 text-gray-800 hover:bg-gray-200`;
        }

      case 'input':
        // Purple styling for input sequence tokens, but can be overridden if matching target/predicted
        if (isTrainingMode && isTargetToken) {
          return `${baseStyles} ${heightStyles} border-2 border-green-500 bg-green-100 text-green-900 font-semibold hover:bg-green-200`;
        } else {
          return `${baseStyles} ${heightStyles} border-2 border-purple-400 bg-purple-100 text-purple-900 hover:bg-purple-200 transition-all hover:bg-red-50 hover:border-red-300`;
        }

      case 'output_prediction':
        return `${baseStyles} ${heightStyles} border-2 border-blue-500 bg-blue-100 text-blue-900 font-semibold`;

      case 'output_target':
        return `${baseStyles} ${heightStyles} border-2 border-green-500 bg-green-100 text-green-900 font-semibold`;

      case 'placeholder':
        return `px-2 sm:px-3 py-1 sm:py-1.5 border border-dashed border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center text-gray-400 italic`;

      default:
        return `${baseStyles} ${heightStyles} border border-gray-300 bg-gray-100 text-gray-800 hover:bg-gray-200`;
    }
  };

  return (
    <div className={getTokenStyles()} onClick={onClick}>
      {text}
      {/* Show embedding as matrix on hover */}
      {showEmbedding && embedding.length > 0 && (
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
          <div className="mb-1 text-gray-600 text-center font-medium">
            {text} embedding
          </div>
          <MatrixDisplay
            data={[embedding]}
            rowLabels={['']}
            columnLabels={Array.from(
              { length: embeddingDimension },
              (_, i) => `d${i + 1}`
            )}
            maxAbsValue={0.2}
            cellSize="xs"
            selectable={false}
            matrixType="none"
          />
        </div>
      )}
    </div>
  );
};

export default Token;
