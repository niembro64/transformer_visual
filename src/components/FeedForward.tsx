import React, { useMemo } from 'react';
import MatrixDisplay from './MatrixDisplay';
import { matrixMultiply, addBias, applyFn, relu } from '../utils/matrixOperations';

interface FeedForwardProps {
  // Input token contexts from attention layer
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
 * Component to visualize the Feed-Forward (MLP) part of a transformer
 * Shows the computation process:
 * 1. First linear transformation: inputs × W1 + b1
 * 2. ReLU activation
 * 3. Second linear transformation: ReLU(inputs × W1 + b1) × W2 + b2
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
  // Number of tokens
  const numTokens = inputs.length;

  // Generate token labels if not provided
  const defaultTokenLabels = useMemo(() => 
    Array.from({ length: numTokens }, (_, i) => `Token ${i+1}`),
    [numTokens]
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
    <div className="flex flex-col gap-6 p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold text-gray-800">Feed-Forward Network (MLP)</h2>
      
      {/* Input from Attention Layer */}
      <div>
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Input from Attention Layer</h3>
        <MatrixDisplay
          data={inputs}
          label="Context Vectors"
          rowLabels={labels}
          maxAbsValue={2}
          className="mb-2"
        />
      </div>

      {/* First Layer and Activation */}
      {showSteps && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">First Layer Output</h4>
            <MatrixDisplay
              data={firstLayerOutput}
              label="First Layer: inputs × W1 + b1"
              rowLabels={labels}
              maxAbsValue={4}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              Linear transformation before activation function.
            </p>
          </div>
          <div>
            <h4 className="text-base font-medium mb-1 text-gray-700">ReLU Activation</h4>
            <MatrixDisplay
              data={activations}
              label="ReLU(First Layer Output)"
              rowLabels={labels}
              maxAbsValue={4}
              cellSize="sm"
            />
            <p className="text-xs text-gray-600 mt-1">
              After ReLU activation (max(0, x)). Negative values are replaced with zeros.
            </p>
          </div>
        </div>
      )}

      {/* Final Output */}
      <div>
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Final Output</h3>
        <MatrixDisplay
          data={output}
          label="activations × W2 + b2"
          rowLabels={labels}
          maxAbsValue={2}
        />
        <p className="text-sm text-gray-600">
          The final output of the feed-forward network. Each token representation has been transformed 
          through a two-layer neural network, allowing it to capture complex relationships.
        </p>
      </div>
      
      {/* Architectural Explanation */}
      <div className="mt-2 p-3 bg-gray-50 rounded-md border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-700 mb-1">About the Feed-Forward Network</h4>
        <p className="text-xs text-gray-600">
          The feed-forward network applies the same transformation to each token independently.
          This allows the model to process the information gathered by the attention mechanism,
          and is crucial for the model's ability to learn complex functions. The network consists of
          two linear transformations with a ReLU activation in between.
        </p>
      </div>
    </div>
  );
};

export default FeedForward;