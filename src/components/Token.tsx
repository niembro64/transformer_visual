import React from 'react';

interface TokenProps {
  text: string;
  isSelected?: boolean;
  onClick?: () => void;
  tokenType: 'tokenizer' | 'input' | 'output_prediction' | 'output_target';
  showEmbedding?: boolean;
  embedding?: number[];
  embeddingDimension?: number;
}

const Token: React.FC<TokenProps> = ({ text, isSelected = false, onClick, tokenType, showEmbedding = false, embedding, embeddingDimension = 0 }) => {
  const getTokenStyles = () => {
    switch (tokenType) {
      case 'tokenizer':
        return 'px-2 sm:px-3 py-1 sm:py-1.5 border border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-gray-100 text-gray-800 hover:bg-gray-200 cursor-pointer font-mono transition-colors';
      case 'input':
        return 'px-2 sm:px-3 py-1 sm:py-1.5 border border-purple-400 bg-purple-100 text-purple-900 hover:bg-purple-200 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] h-7 sm:h-9 text-center shadow-sm font-mono cursor-pointer transition-all';
      case 'output_prediction':
        return 'px-2 sm:px-3 py-1 sm:py-1.5 border-2 border-blue-500 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-blue-100 font-mono cursor-pointer text-blue-900 font-semibold';
      case 'output_target':
        return 'px-2 sm:px-3 py-1 sm:py-1.5 border-2 border-green-500 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-green-100 font-mono cursor-pointer text-green-900 font-semibold';
      default:
        return 'px-2 sm:px-3 py-1 sm:py-1.5 border border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-gray-100 text-gray-800 hover:bg-gray-200 cursor-pointer font-mono transition-colors';
    }
  };

  return (
    <div
      className={`${getTokenStyles()} group relative`}
      onClick={onClick}
    >
      {text}
    </div>
  );
};

export default Token;
