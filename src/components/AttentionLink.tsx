import React from 'react';
import { Attention } from '../types';

interface AttentionLinkProps {
  attention: Attention;
  sourcePositions: Record<number, { x: number; y: number }>;
  targetPositions: Record<number, { x: number; y: number }>;
}

const AttentionLink: React.FC<AttentionLinkProps> = ({ 
  attention, 
  sourcePositions, 
  targetPositions 
}) => {
  const sourcePos = sourcePositions[attention.from];
  const targetPos = targetPositions[attention.to];
  
  if (!sourcePos || !targetPos) return null;
  
  // Calculate opacity based on weight (0 to 1)
  const opacity = Math.max(0.1, Math.min(0.9, attention.weight));
  const strokeWidth = Math.max(1, Math.min(5, attention.weight * 5));
  
  return (
    <line
      x1={sourcePos.x}
      y1={sourcePos.y}
      x2={targetPos.x}
      y2={targetPos.y}
      stroke="rgba(59, 130, 246, var(--tw-text-opacity))"
      strokeWidth={strokeWidth}
      style={{ '--tw-text-opacity': opacity } as React.CSSProperties}
    />
  );
};

export default AttentionLink;