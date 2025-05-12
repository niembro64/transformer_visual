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
  // Currently selected element coordinates [row, col] or null if none selected
  selectedElement?: [number, number] | null;
  // Callback when an element is clicked in the embedding matrix
  onElementClick?: (row: number, col: number) => void;
  // Callback when element value changes via slider
  onValueChange?: (newValue: number) => void;
}

/**
 * Component to visualize a single Self-Attention head in a transformer model
 * Shows the full attention computation process:
 * 1. Project input embeddings to Query, Key, and Value matrices
 * 2. Compute scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T/√d_k)V
 * 3. Produce output embeddings through attention weighted aggregation
 */
const AttentionHead: React.FC<AttentionHeadProps> = ({
  embeddings,
  weightQ,
  weightK,
  weightV,
  tokenLabels,
  showSteps = true,
  onContextComputed,
  selectedElement = null,
  onElementClick,
  onValueChange
}) => {
  // Number of tokens and dimensionality
  const numTokens = embeddings.length;
  const modelDim = embeddings[0].length;
  const headDim = weightQ[0].length;

  // Project input embeddings to Query, Key, and Value matrices
  const Q = useMemo(() => matrixMultiply(embeddings, weightQ), [embeddings, weightQ]);
  const K = useMemo(() => matrixMultiply(embeddings, weightK), [embeddings, weightK]);
  const V = useMemo(() => matrixMultiply(embeddings, weightV), [embeddings, weightV]);

  // Compute scaled dot-product attention
  const Kt = useMemo(() => transpose(K), [K]);
  const attentionScores = useMemo(() => {
    const scores = matrixMultiply(Q, Kt);
    // Scale by square root of the attention dimension (d_k)
    return scaleMatrix(scores, 1 / Math.sqrt(headDim));
  }, [Q, Kt, headDim]);

  // Apply softmax to get attention weights
  const attentionWeights = useMemo(() => softmax(attentionScores), [attentionScores]);

  // Compute output as attention-weighted sum of values
  const attentionOutput = useMemo(() => matrixMultiply(attentionWeights, V), [attentionWeights, V]);

  // Use useEffect to call the callback after render
  useEffect(() => {
    if (onContextComputed && attentionOutput.length > 0) {
      onContextComputed(attentionOutput);
    }
  }, [attentionOutput, onContextComputed]);

  // Generate token labels if not provided
  const defaultTokenLabels = useMemo(() => 
    Array.from({ length: numTokens }, (_, i) => `Token ${i+1}`),
    [numTokens]
  );
  
  // Generate feature labels for the model dimensions (d_model)
  const modelDimLabels = useMemo(() => 
    Array.from({ length: modelDim }, (_, i) => `d_${i+1}`),
    [modelDim]
  );
  
  // Generate feature labels for the attention head dimensions (d_k, d_v)
  const headDimLabels = useMemo(() => 
    Array.from({ length: headDim }, (_, i) => `d_${i+1}`),
    [headDim]
  );
  
  const labels = tokenLabels || defaultTokenLabels;

  return (
    <div className="flex flex-col gap-1 p-1 bg-white rounded">
      {/* Main Container - Horizontal Layout */}
      <div className="grid grid-cols-12 gap-2">
        {/* Left Column: Input Embeddings */}
        <div className="col-span-3">
          <h3 className="text-sm font-semibold mb-1 text-gray-700">Input Embeddings</h3>
          <MatrixDisplay
            data={embeddings}
            label="Tokens"
            rowLabels={labels}
            columnLabels={modelDimLabels}
            maxAbsValue={0.2}
            className="mb-1"
            selectable={true}
            selectedElement={selectedElement}
            onElementClick={onElementClick}
            onValueChange={onValueChange}
          />
        </div>

        {/* Middle Column: Projection Matrices */}
        {showSteps && (
          <div className="col-span-3">
            <h3 className="text-sm font-semibold mb-1 text-gray-700">Weights</h3>
            <div className="grid grid-cols-3 gap-1">
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">W^Q</h4>
                <MatrixDisplay
                  data={weightQ}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="sm"
                />
              </div>
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">W^K</h4>
                <MatrixDisplay
                  data={weightK}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="sm"
                />
              </div>
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">W^V</h4>
                <MatrixDisplay
                  data={weightV}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="sm"
                />
              </div>
            </div>
          </div>
        )}

        {/* Right Column: Q, K, V Matrices */}
        {showSteps && (
          <div className="col-span-6">
            <h3 className="text-sm font-semibold mb-1 text-gray-700">Projected Matrices</h3>
            <div className="grid grid-cols-3 gap-1">
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">Q</h4>
                <MatrixDisplay
                  data={Q}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="sm"
                />
              </div>
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">K</h4>
                <MatrixDisplay
                  data={K}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="sm"
                />
              </div>
              <div>
                <h4 className="text-xs font-medium mb-1 text-center text-gray-700">V</h4>
                <MatrixDisplay
                  data={V}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="sm"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Second Row: Attention Scores, Weights, and Output Calculation */}
      {showSteps && (
        <div className="grid grid-cols-12 gap-2 mt-1">
          {/* Left Column: Attention Scores */}
          <div className="col-span-3">
            <h3 className="text-xs font-semibold mb-1 text-gray-700">Attention Scores</h3>
            <div>
              <h4 className="text-xs font-medium mb-1 text-center text-gray-700">QK^T/√d_k</h4>
              <MatrixDisplay
                data={attentionScores}
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="sm"
              />
            </div>
          </div>

          {/* Middle Column: Attention Weights */}
          <div className="col-span-3">
            <h3 className="text-xs font-semibold mb-1 text-gray-700">Attention Weights</h3>
            <div>
              <h4 className="text-xs font-medium mb-1 text-center text-gray-700">softmax</h4>
              <MatrixDisplay
                data={attentionWeights}
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="sm"
              />
            </div>
          </div>

          {/* Right Column: Final Output */}
          <div className="col-span-6">
            <h3 className="text-xs font-semibold mb-1 text-gray-700">Output</h3>
            <div>
              <h4 className="text-xs font-medium mb-1 text-center text-gray-700">Attention Output</h4>
              <MatrixDisplay
                data={attentionOutput}
                rowLabels={labels}
                columnLabels={headDimLabels}
                maxAbsValue={0.3}
                cellSize="sm"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AttentionHead;