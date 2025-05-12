import React, { useMemo, useEffect } from 'react';
import MatrixDisplay from './MatrixDisplay';
import {
  matrixMultiply,
  transpose,
  softmax,
  scaleMatrix
} from '../utils/matrixOperations';

interface AttentionHeadProps {
  // Input token embeddings (one array per token)
  embeddings: number[][];
  // Weight matrices for transforming embeddings to Q, K, V
  weightQ: number[][];
  weightK: number[][];
  weightV: number[][];
  // Token labels (optional)
  tokenLabels?: string[];
  // Whether to show intermediate computation steps
  showSteps?: boolean;
  // Optional callback when context vectors are computed
  onContextComputed?: (context: number[][]) => void;
}

/**
 * Component to visualize a single attention head in a transformer model
 * Shows the full attention computation process:
 * 1. Project input embeddings to queries (Q), keys (K), and values (V)
 * 2. Compute attention scores as Q·Kᵀ/√d
 * 3. Apply softmax to get attention weights
 * 4. Compute weighted sum of values to get context vectors
 */
const AttentionHead: React.FC<AttentionHeadProps> = ({
  embeddings,
  weightQ,
  weightK,
  weightV,
  tokenLabels,
  showSteps = true,
  onContextComputed
}) => {
  // Number of tokens and dimensionality
  const numTokens = embeddings.length;
  const headDim = weightQ[0].length;

  // Compute the Q, K, V projections
  const Q = useMemo(() => matrixMultiply(embeddings, weightQ), [embeddings, weightQ]);
  const K = useMemo(() => matrixMultiply(embeddings, weightK), [embeddings, weightK]);
  const V = useMemo(() => matrixMultiply(embeddings, weightV), [embeddings, weightV]);

  // Compute attention scores: Q·Kᵀ/√d
  const Kt = useMemo(() => transpose(K), [K]);
  const attentionScores = useMemo(() => {
    const scores = matrixMultiply(Q, Kt);
    // Scale by square root of the head dimension
    return scaleMatrix(scores, 1 / Math.sqrt(headDim));
  }, [Q, Kt, headDim]);

  // Apply softmax to get normalized attention weights
  const attentionWeights = useMemo(() => softmax(attentionScores), [attentionScores]);

  // Compute context vectors as weighted sum of values
  const context = useMemo(() => matrixMultiply(attentionWeights, V), [attentionWeights, V]);

  // Use useEffect to call the callback after render
  useEffect(() => {
    if (onContextComputed && context.length > 0) {
      onContextComputed(context);
    }
  }, [context, onContextComputed]);

  // Generate token labels if not provided
  const defaultTokenLabels = useMemo(() => 
    Array.from({ length: numTokens }, (_, i) => `Token ${i+1}`),
    [numTokens]
  );
  
  const labels = tokenLabels || defaultTokenLabels;

  return (
    <div className="flex flex-col gap-6 p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold text-gray-800">Attention Head Visualization</h2>
      
      {/* Input Embeddings */}
      <div>
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Input Token Embeddings</h3>
        <MatrixDisplay
          data={embeddings}
          label="Input Embeddings"
          rowLabels={labels}
          maxAbsValue={2}
          className="mb-2"
        />
        <p className="text-sm text-gray-600">
          Each row represents an embedding vector for a token in the sequence.
        </p>
      </div>

      {/* Q, K, V Projections */}
      {showSteps && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Query Projection (Q)</h4>
            <MatrixDisplay
              data={Q}
              label="Q = Embeddings × W_Q"
              rowLabels={labels}
              maxAbsValue={2}
              cellSize="sm"
            />
          </div>
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Key Projection (K)</h4>
            <MatrixDisplay
              data={K}
              label="K = Embeddings × W_K"
              rowLabels={labels}
              maxAbsValue={2}
              cellSize="sm"
            />
          </div>
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Value Projection (V)</h4>
            <MatrixDisplay
              data={V}
              label="V = Embeddings × W_V"
              rowLabels={labels}
              maxAbsValue={2}
              cellSize="sm"
            />
          </div>
        </div>
      )}

      {/* Attention Scores and Weights */}
      {showSteps && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Attention Scores</h4>
            <MatrixDisplay
              data={attentionScores}
              label={`Q·Kᵀ/√${headDim} (Raw Scores)`}
              rowLabels={labels}
              columnLabels={labels}
              maxAbsValue={3}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Raw attention scores before softmax. Each cell shows how much token i attends to token j.
            </p>
          </div>
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Attention Weights</h4>
            <MatrixDisplay
              data={attentionWeights}
              label="softmax(Attention Scores)"
              rowLabels={labels}
              columnLabels={labels}
              maxAbsValue={1}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Normalized attention weights (sum to 1 across each row). Brighter blue indicates stronger attention.
            </p>
          </div>
        </div>
      )}

      {/* Output Context Vectors */}
      <div>
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Output Context Vectors</h3>
        <MatrixDisplay
          data={context}
          label="Context = Attention Weights × V"
          rowLabels={labels}
          maxAbsValue={2}
        />
        <p className="text-sm text-gray-600">
          The final output of the attention head. Each token now contains information from other relevant tokens in the sequence.
        </p>
      </div>
    </div>
  );
};

export default AttentionHead;