import React from 'react';

interface TokenProps {
  text: string;
  isSelected?: boolean;
  onClick?: () => void;
  tokenType: 'token_training' | 'token_input' | 'token_output';
  showEmbedding?: boolean;
  embedding?: number[];
  embeddingDimension?: number;
}

const Token: React.FC<TokenProps> = ({ text, isSelected = false, onClick, tokenType, showEmbedding = false, embedding, embeddingDimension = 0 }) => {
  const getTokenStyles = () => {
    if (isSelected) {
      return 'bg-blue-500 text-white shadow-lg transform scale-110';
    }
    
    switch (tokenType) {
      case 'token_training':
        return 'bg-blue-100 text-blue-800 border border-blue-300 hover:bg-blue-200';
      case 'token_output':
        return 'bg-green-100 text-green-800 border border-green-300 hover:bg-green-200';
      case 'token_input':
      default:
        return 'bg-gray-100 text-gray-800 hover:bg-gray-200';
    }
  };

  return (
    <div
      className={`px-3 py-2 rounded-md cursor-pointer transition-all ${getTokenStyles()}`}
      onClick={onClick}
    >
      {text}
    </div>
  );
};

export default Token;
