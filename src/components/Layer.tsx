import React, { useState, useRef, useEffect, useCallback } from 'react';
import { LayerData } from '../types';
import Token from './Token';
import AttentionLink from './AttentionLink';

interface LayerProps {
  layer: LayerData;
}

const Layer: React.FC<LayerProps> = ({ layer }) => {
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [tokenPositions, setTokenPositions] = useState<Record<number, { x: number; y: number }>>({});
  const tokensRef = useRef<(HTMLDivElement | null)[]>([]);
  
  // Set up refs array for tokens
  if (tokensRef.current.length !== layer.tokens.length) {
    tokensRef.current = Array(layer.tokens.length).fill(null);
  }
  
  // Calculate token positions for drawing attention lines
  useEffect(() => {
    const newPositions: Record<number, { x: number; y: number }> = {};
    
    tokensRef.current.forEach((ref, index) => {
      if (ref) {
        const rect = ref.getBoundingClientRect();
        const tokenId = layer.tokens[index].id;
        newPositions[tokenId] = {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2
        };
      }
    });
    
    setTokenPositions(newPositions);
  }, [layer.tokens]);
  
  // Filter attentions for selected token
  const relevantAttentions = selectedToken !== null
    ? layer.attentions.filter(att => att.from === selectedToken || att.to === selectedToken)
    : [];
  
  // Callback ref for setting refs in the array
  const setTokenRef = useCallback((el: HTMLDivElement | null, idx: number) => {
    tokensRef.current[idx] = el;
  }, []);
  
  return (
    <div className="mb-10">
      <h3 className="text-xl font-bold mb-2">{layer.name}</h3>
      
      {/* Tokens */}
      <div className="flex flex-wrap gap-2 mb-4">
        {layer.tokens.map((token, idx) => (
          <div 
            key={token.id} 
            ref={(el) => setTokenRef(el, idx)}
          >
            <Token
              text={token.text}
              isSelected={selectedToken === token.id}
              onClick={() => setSelectedToken(token.id === selectedToken ? null : token.id)}
              tokenType="input"
            />
          </div>
        ))}
      </div>
      
      {/* SVG overlay for attention lines */}
      {selectedToken !== null && (
        <svg 
          className="absolute top-0 left-0 w-full h-full pointer-events-none z-10"
          style={{ position: 'fixed' }}
        >
          {relevantAttentions.map((attention, idx) => (
            <AttentionLink
              key={idx}
              attention={attention}
              sourcePositions={tokenPositions}
              targetPositions={tokenPositions}
            />
          ))}
        </svg>
      )}
    </div>
  );
};

export default Layer;