import React from 'react';
import { TokenData } from '../types';

interface TokenProps {
  token: TokenData;
  isSelected: boolean;
  onClick: (tokenId: number) => void;
  tokenType: 'token_training' | 'token_input' | 'token_output';
}

const Token: React.FC<TokenProps> = ({ token, isSelected, onClick }) => {
  return (
    <div
      className={`
        px-3 py-2 rounded-md cursor-pointer transition-all
        ${
          isSelected
            ? 'bg-blue-500 text-white shadow-lg transform scale-110'
            : 'bg-gray-100 hover:bg-gray-200'
        }
      `}
      onClick={() => onClick(token.id)}
    >
      {token.text}
    </div>
  );
};

export default Token;
