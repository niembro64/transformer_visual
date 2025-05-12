import React, { useMemo } from 'react';
import MatrixDisplay from './MatrixDisplay';
import { matrixMultiply, addBias, applyFn, relu } from '../utils/matrixOperations';

interface FeedForwardProps {
  // Input token representations from attention layer
  inputs: number[][];
  // First layer weights and biases
  W1: number[][];
  b1: number[];
  // Second layer weights and biases
  W2: number[][];
  b2: number[];
  // Optional token labels
  tokenLabels?: string[];
  // Whether to show intermediate results
  showSteps?: boolean;
}

/**
 * Component to visualize the Position-wise Feed-Forward Network (FFN) part of a transformer
 * Shows the computation process:
 * FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
 * As described in "Attention Is All You Need" paper (Vaswani et al., 2017)
 */
const FeedForward: React.FC<FeedForwardProps> = ({
  inputs,
  W1,
  b1,
  W2,
  b2,
  tokenLabels,
  showSteps = true
}) => {
  // Number of tokens and dimensionality
  const numTokens = inputs.length;
  const d_model = inputs[0].length;    // Input/output dimension (d_model)
  const d_ff = W1[0].length;           // Inner dimension of FFN (d_ff)

  // Generate token labels if not provided
  const defaultTokenLabels = useMemo(() => 
    Array.from({ length: numTokens }, (_, i) => `Token ${i+1}`),
    [numTokens]
  );
  
  // Generate feature labels for different dimensions
  const modelDimLabels = useMemo(() => 
    Array.from({ length: d_model }, (_, i) => `d_${i+1}`),
    [d_model]
  );
  
  const ffnDimLabels = useMemo(() => 
    Array.from({ length: d_ff }, (_, i) => `ff_${i+1}`),
    [d_ff]
  );
  
  const labels = tokenLabels || defaultTokenLabels;

  // First layer computation: inputs × W1 + b1
  const firstLayerOutput = useMemo(() => {
    const product = matrixMultiply(inputs, W1);
    return addBias(product, b1);
  }, [inputs, W1, b1]);

  // Apply ReLU activation function
  const activations = useMemo(() => 
    applyFn(firstLayerOutput, relu),
    [firstLayerOutput]
  );

  // Second layer computation: activations × W2 + b2
  const output = useMemo(() => {
    const product = matrixMultiply(activations, W2);
    return addBias(product, b2);
  }, [activations, W2, b2]);

  return (
    <div className="flex flex-col gap-3 p-2 bg-white rounded">
      <h2 className="text-lg font-bold text-gray-800">Feed-Forward Network</h2>
      
      {/* Input from Attention Layer */}
      <div>
        <h3 className="text-base font-semibold mb-1 text-gray-700">Attention Output</h3>
        <MatrixDisplay
          data={inputs}
          label="Attention Output"
          rowLabels={labels}
          columnLabels={modelDimLabels}
          maxAbsValue={0.2}
          className="mb-2"
        />
        <p className="text-xs text-gray-600">
          Token representations after attention (d_model = {d_model}).
        </p>
      </div>

      {/* Learned Weight Matrices */}
      {showSteps && (
        <div className="mb-6">
          <h3 className="text-base font-semibold mb-1 text-gray-700">FFN Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">First Linear Layer (W₁)</h4>
              <MatrixDisplay
                data={W1}
                label="W₁"
                rowLabels={modelDimLabels}
                columnLabels={ffnDimLabels}
                maxAbsValue={0.1}
                cellSize="sm"
              />
              <div className="mt-3">
                <h4 className="text-base font-medium mb-1 text-gray-700">First Layer Bias (b₁)</h4>
                <MatrixDisplay
                  data={[b1]}
                  label="b₁"
                  rowLabels={["bias"]}
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.05}
                  cellSize="sm"
                />
              </div>
              <p className="text-xs text-gray-600 mt-1">
                Projects from model dimension (d_model = {d_model}) to inner FFN dimension (d_ff = {d_ff}).
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">Second Linear Layer (W₂)</h4>
              <MatrixDisplay
                data={W2}
                label="W₂"
                rowLabels={ffnDimLabels}
                columnLabels={modelDimLabels}
                maxAbsValue={0.1}
                cellSize="sm"
              />
              <div className="mt-3">
                <h4 className="text-base font-medium mb-1 text-gray-700">Second Layer Bias (b₂)</h4>
                <MatrixDisplay
                  data={[b2]}
                  label="b₂"
                  rowLabels={["bias"]}
                  columnLabels={modelDimLabels}
                  maxAbsValue={0.05}
                  cellSize="sm"
                />
              </div>
              <p className="text-xs text-gray-600 mt-1">
                Projects back from inner FFN dimension (d_ff = {d_ff}) to model dimension (d_model = {d_model}).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* First Layer and Activation */}
      {showSteps && (
        <div>
          <h3 className="text-base font-semibold mb-1 text-gray-700">FFN Computation Steps</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">First Linear Transformation</h4>
              <MatrixDisplay
                data={firstLayerOutput}
                label="xW₁ + b₁"
                rowLabels={labels}
                columnLabels={ffnDimLabels}
                maxAbsValue={0.5}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Output after the first linear transformation (d_ff = {d_ff}).
              </p>
            </div>
            <div>
              <h4 className="text-base font-medium mb-1 text-gray-700">After ReLU Activation</h4>
              <MatrixDisplay
                data={activations}
                label="max(0, xW₁ + b₁)"
                rowLabels={labels}
                columnLabels={ffnDimLabels}
                maxAbsValue={0.5}
                cellSize="sm"
              />
              <p className="text-xs text-gray-600 mt-1">
                Output after applying ReLU activation function. Negative values are replaced with zeros.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Final Output */}
      <div className="mt-2">
        <h3 className="text-base font-semibold mb-1 text-gray-700">FFN Output</h3>
        <MatrixDisplay
          data={output}
          label="FFN(x) = max(0, xW₁ + b₁)W₂ + b₂"
          rowLabels={labels}
          columnLabels={modelDimLabels}
          maxAbsValue={0.2}
        />
      </div>
      
      {/* Architectural Explanation */}
    </div>
  );
};

export default FeedForward;