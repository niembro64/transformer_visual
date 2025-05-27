import React, { useMemo, useEffect } from 'react';
import MatrixDisplay from './MatrixDisplay';
import {
  matrixMultiply,
  transpose,
  softmax,
  scaleMatrix,
  isPortraitOrientation,
  hasInvalidValues,
  getMatrixErrorDetails
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
  // Currently selected element coordinates
  selectedElement?: {
    matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'weightW1' | 'weightW2' | 'none';
    row: number;
    col: number;
  } | null;
  // Callback when an element is clicked in any matrix
  onElementClick?: (matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'weightW1' | 'weightW2' | 'none', row: number, col: number) => void;
  // Callback when element value changes via slider
  onValueChange?: (newValue: number) => void;
  // Label for value editing
  valueLabel?: string;
  // Callback when calculation error occurs
  onCalculationError?: (error: string | null) => void;
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
  onValueChange,
  valueLabel,
  onCalculationError
}) => {
  // Number of tokens and dimensionality
  const numTokens = embeddings.length;
  const modelDim = embeddings[0].length;
  const headDim = weightQ[0].length;

  // Project input embeddings to Query, Key, and Value matrices - memoized with proper dependencies
  const Q = useMemo(() => {
    try {
      const result = matrixMultiply(embeddings, weightQ);
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'Query matrix (Q)');
        onCalculationError?.(error || 'Invalid values in Query matrix');
        return embeddings; // Fallback
      }
      return result;
    } catch (error) {
      onCalculationError?.(`Error computing Q: ${error}`);
      return embeddings;
    }
  }, [embeddings, weightQ, onCalculationError]);

  const K = useMemo(() => {
    try {
      const result = matrixMultiply(embeddings, weightK);
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'Key matrix (K)');
        onCalculationError?.(error || 'Invalid values in Key matrix');
        return embeddings; // Fallback
      }
      return result;
    } catch (error) {
      onCalculationError?.(`Error computing K: ${error}`);
      return embeddings;
    }
  }, [embeddings, weightK, onCalculationError]);

  const V = useMemo(() => {
    try {
      const result = matrixMultiply(embeddings, weightV);
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'Value matrix (V)');
        onCalculationError?.(error || 'Invalid values in Value matrix');
        return embeddings; // Fallback
      }
      return result;
    } catch (error) {
      onCalculationError?.(`Error computing V: ${error}`);
      return embeddings;
    }
  }, [embeddings, weightV, onCalculationError]);

  // Compute scaled dot-product attention
  const Kt = useMemo(() => transpose(K), [K]);
  
  const attentionScores = useMemo(() => {
    try {
      const scores = matrixMultiply(Q, Kt);
      const scaledScores = scaleMatrix(scores, 1 / Math.sqrt(headDim));
      
      if (hasInvalidValues(scaledScores)) {
        const error = getMatrixErrorDetails(scaledScores, 'Attention scores');
        onCalculationError?.(error || 'Invalid values in attention scores');
        // Return identity-like matrix as fallback
        return Q.map((_, i) => Q.map((_, j) => i === j ? 1 : 0));
      }
      
      return scaledScores;
    } catch (error) {
      onCalculationError?.(`Error computing attention scores: ${error}`);
      return Q.map((_, i) => Q.map((_, j) => i === j ? 1 : 0));
    }
  }, [Q, Kt, headDim, onCalculationError]);

  // Apply softmax to get attention weights
  const attentionWeights = useMemo(() => {
    try {
      const result = softmax(attentionScores);
      
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'Attention weights (after softmax)');
        onCalculationError?.(error || 'Invalid values in attention weights');
        // Return uniform attention as fallback
        const numTokens = attentionScores.length;
        return attentionScores.map(() => Array(numTokens).fill(1 / numTokens));
      }
      
      // Clear any previous errors if calculation succeeded
      onCalculationError?.(null);
      return result;
    } catch (error) {
      onCalculationError?.(`Error computing softmax: ${error}`);
      const numTokens = attentionScores.length;
      return attentionScores.map(() => Array(numTokens).fill(1 / numTokens));
    }
  }, [attentionScores, onCalculationError]);

  // Compute output as attention-weighted sum of values
  const rawAttentionOutput = useMemo(() => {
    try {
      const result = matrixMultiply(attentionWeights, V);
      
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'Attention output');
        onCalculationError?.(error || 'Invalid values in attention output');
        return V; // Use V as fallback
      }
      
      return result;
    } catch (error) {
      onCalculationError?.(`Error computing attention output: ${error}`);
      return V;
    }
  }, [attentionWeights, V, onCalculationError]);
  
  // Use raw attention output directly (no dropout)
  const attentionOutput = rawAttentionOutput;

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
    <div className="flex flex-col gap-0.5 p-0.5 bg-white rounded">
      {/* Main Container - Responsive layout based on orientation */}
      <div className={`flex flex-col ${!isPortraitOrientation() ? 'md:grid md:grid-cols-2' : ''} lg:grid lg:grid-cols-9 gap-3 lg:gap-1`}>
        {/* Projection Matrices - Full width in portrait, 1/2 in landscape on mobile, 4/9 on desktop */}
        {showSteps && (
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-4 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Weights</h3>
            <div className={`${isPortraitOrientation() ? 'grid grid-cols-3' : 'grid grid-cols-3'} gap-0.5 w-full`}>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">W^Q</h4>
                <MatrixDisplay
                  data={weightQ}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="xs"
                  selectable={true}
                  selectedElement={selectedElement?.matrixType === 'weightQ' ? selectedElement : null}
                  matrixType="weightQ"
                  onElementClick={onElementClick}
                  onValueChange={onValueChange}
                  valueLabel={selectedElement?.matrixType === 'weightQ' ? valueLabel : undefined}
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">W^K</h4>
                <MatrixDisplay
                  data={weightK}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="xs"
                  selectable={true}
                  selectedElement={selectedElement?.matrixType === 'weightK' ? selectedElement : null}
                  matrixType="weightK"
                  onElementClick={onElementClick}
                  onValueChange={onValueChange}
                  valueLabel={selectedElement?.matrixType === 'weightK' ? valueLabel : undefined}
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">W^V</h4>
                <MatrixDisplay
                  data={weightV}
                  rowLabels={modelDimLabels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.5}
                  cellSize="xs"
                  selectable={true}
                  selectedElement={selectedElement?.matrixType === 'weightV' ? selectedElement : null}
                  matrixType="weightV"
                  onElementClick={onElementClick}
                  onValueChange={onValueChange}
                  valueLabel={selectedElement?.matrixType === 'weightV' ? valueLabel : undefined}
                />
              </div>
            </div>
          </div>
        )}

        {/* Q, K, V Matrices - Full width in portrait, 1/2 in landscape on mobile, 5/9 on desktop */}
        {showSteps && (
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-5 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Projected Matrices</h3>
            <div className={`${isPortraitOrientation() ? 'grid grid-cols-3' : 'grid grid-cols-3'} gap-0.5 w-full`}>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">Q</h4>
                <MatrixDisplay
                  data={Q}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="xs"
                  selectable={false}
                  matrixType="none"
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">K</h4>
                <MatrixDisplay
                  data={K}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="xs"
                  selectable={false}
                  matrixType="none"
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">V</h4>
                <MatrixDisplay
                  data={V}
                  rowLabels={labels}
                  columnLabels={headDimLabels}
                  maxAbsValue={0.3}
                  cellSize="xs"
                  selectable={false}
                  matrixType="none"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Second Row: Attention Scores, Weights, and Output Calculation - Responsive layout based on orientation */}
      {showSteps && (
        <div className={`flex flex-col ${!isPortraitOrientation() ? 'md:grid md:grid-cols-3' : ''} lg:grid lg:grid-cols-12 gap-3 lg:gap-1 mt-3 lg:mt-0.5`}>
          {/* Attention Scores - Full width in portrait, 1/3 in landscape on mobile, 3/12 on desktop */}
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-3 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Attention Scores</h3>
            <div className="flex flex-col items-center justify-center w-full">
              <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">QK^T/√d_k</h4>
              <MatrixDisplay
                data={attentionScores}
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="xs"
                selectable={false}
                matrixType="none"
              />
            </div>
          </div>

          {/* Attention Weights - Full width in portrait, 1/3 in landscape on mobile, 3/12 on desktop */}
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-3 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Attention Weights</h3>
            <div className="flex flex-col items-center justify-center w-full">
              <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">softmax</h4>
              <MatrixDisplay
                data={attentionWeights}
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="xs"
                selectable={false}
                matrixType="none"
              />
            </div>
          </div>

          {/* Final Output - Full width in portrait, 1/3 in landscape on mobile, 6/12 on desktop */}
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-6 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Attention Output</h3>
            <div className="flex flex-col items-center justify-center w-full">
              <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">
                Output
              </h4>
              <MatrixDisplay
                data={attentionOutput}
                rowLabels={labels}
                columnLabels={headDimLabels}
                maxAbsValue={0.3}
                cellSize="xs"
                selectable={false}
                matrixType="none"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AttentionHead;