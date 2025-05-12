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
  onElementClick
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
    <div className="flex flex-col gap-6 p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold text-gray-800">Self-Attention Mechanism</h2>
      
      {/* Input Embeddings */}
      <div>
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Input Token Embeddings (X)</h3>
        <MatrixDisplay
          data={embeddings}
          label="Input Embeddings (d_model) - Click an element to edit"
          rowLabels={labels}
          columnLabels={modelDimLabels}
          maxAbsValue={0.2}
          className="mb-2"
          selectable={true}
          selectedElement={selectedElement}
          onElementClick={onElementClick}
        />
        <p className="text-sm text-gray-600">
          Each row represents the embedding vector for a token (d_model = {modelDim}).
        </p>
      </div>

      {/* Learned Projection Matrices */}
      {showSteps && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2 text-gray-700">Linear Projection Matrices</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Query Projection (W^Q)</h4>
              <MatrixDisplay
                data={weightQ}
                label="W^Q"
                rowLabels={modelDimLabels}
                columnLabels={headDimLabels}
                maxAbsValue={0.5}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Projects embeddings from d_model → d_k dimension.
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Key Projection (W^K)</h4>
              <MatrixDisplay
                data={weightK}
                label="W^K"
                rowLabels={modelDimLabels}
                columnLabels={headDimLabels}
                maxAbsValue={0.5}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Projects embeddings from d_model → d_k dimension.
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Value Projection (W^V)</h4>
              <MatrixDisplay
                data={weightV}
                label="W^V"
                rowLabels={modelDimLabels}
                columnLabels={headDimLabels}
                maxAbsValue={0.5}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Projects embeddings from d_model → d_v dimension.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Query, Key, Value Matrices */}
      {showSteps && (
        <div>
          <h3 className="text-lg font-semibold mb-2 text-gray-700">Query, Key, Value Matrices</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Query Matrix (Q)</h4>
              <MatrixDisplay
                data={Q}
                label="Q = X × W^Q"
                rowLabels={labels}
                columnLabels={headDimLabels}
                maxAbsValue={0.3}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Queries for each token (d_k = {headDim}).
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Key Matrix (K)</h4>
              <MatrixDisplay
                data={K}
                label="K = X × W^K"
                rowLabels={labels}
                columnLabels={headDimLabels}
                maxAbsValue={0.3}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Keys for each token (d_k = {headDim}).
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Value Matrix (V)</h4>
              <MatrixDisplay
                data={V}
                label="V = X × W^V"
                rowLabels={labels}
                columnLabels={headDimLabels}
                maxAbsValue={0.3}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Values for each token (d_v = {headDim}).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Attention Scores and Weights */}
      {showSteps && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2 text-gray-700">Scaled Dot-Product Attention</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Scaled Dot-Product</h4>
              <MatrixDisplay
                data={attentionScores}
                label={`QK^T/√d_k`}
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Compatibility scores between query and key pairs, scaled by 1/√{headDim}.
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Attention Weights</h4>
              <MatrixDisplay
                data={attentionWeights}
                label="softmax(QK^T/√d_k)"
                rowLabels={labels}
                columnLabels={labels}
                maxAbsValue={1.0}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Normalized attention probabilities that sum to 1 across each row.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Attention Output Calculation */}
      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Attention Output Calculation</h3>
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 mb-4">
          <h4 className="text-base font-medium mb-2 text-blue-800">How Attention Output Is Calculated:</h4>
          <p className="text-sm text-blue-700 leading-relaxed">
            The attention output is calculated by multiplying the attention weights matrix by the Value matrix:
          </p>
          <div className="flex flex-col items-center my-2">
            <div className="font-mono text-blue-900 text-sm bg-blue-100 px-3 py-1 rounded mb-1">
              Attention Output = Attention Weights × V
            </div>
            <div className="font-mono text-blue-900 text-sm bg-blue-100 px-3 py-1 rounded">
              Attention Output = softmax(QK^T/√d_k) × V
            </div>
          </div>
          <p className="text-sm text-blue-700 mt-2">
            For each token (row) in the output:
          </p>
          <ol className="list-decimal list-inside text-sm text-blue-700 mt-1 space-y-1">
            <li>Its attention weights (one row from the attention weights matrix) determine how much information to gather from each token</li>
            <li>Each weight is multiplied by the corresponding token's value vector (row in the V matrix)</li>
            <li>These weighted value vectors are summed to produce the output for that token</li>
          </ol>
          <p className="text-sm text-blue-700 mt-2">
            This means each output token is a weighted combination of all value vectors, with weights determined by attention.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Attention Weights</h4>
            <MatrixDisplay
              data={attentionWeights}
              label="Attention Weights"
              rowLabels={labels}
              columnLabels={labels}
              maxAbsValue={1.0}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Each row contains weights that sum to 1, determining how much each token attends to every other token.
            </p>
          </div>
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">Value Matrix (V)</h4>
            <MatrixDisplay
              data={V}
              label="V"
              rowLabels={labels}
              columnLabels={headDimLabels}
              maxAbsValue={0.3}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Contains the value vectors that will be weighted and aggregated according to attention weights.
            </p>
          </div>
        </div>
      </div>

      {/* Final Attention Output */}
      <div className="mt-4">
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Final Attention Output</h3>
        <MatrixDisplay
          data={attentionOutput}
          label="Attention(Q,K,V) = softmax(QK^T/√d_k) × V"
          rowLabels={labels}
          columnLabels={headDimLabels}
          maxAbsValue={0.3}
        />
        <p className="text-sm text-gray-600">
          The final output of the self-attention mechanism. Each row is a weighted combination of value vectors,
          where the weights come from the corresponding row in the attention weights matrix.
          This allows each token to gather relevant information from all other tokens in the sequence.
        </p>
      </div>
      
      {/* Mathematical Intuition */}
      <div className="mt-4 p-3 bg-gray-50 rounded-md border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Attention Mechanism Intuition</h4>
        <p className="text-xs text-gray-600">
          The self-attention mechanism allows each token to query all other tokens for relevant information.
          Tokens with higher attention scores contribute more of their value vectors to the output.
          This weighted aggregation enables the model to focus on relevant parts of the input sequence
          when producing the output representation for each token.
        </p>
      </div>
    </div>
  );
};

export default AttentionHead;