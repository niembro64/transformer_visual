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
    <div className="flex flex-col gap-0.5 p-0.5 bg-white rounded">
      {/* Main Container - Horizontal Layout */}
      <div className="grid grid-cols-12 gap-1">
        {/* Left Column: Input */}
        <div className="col-span-3">
          <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700">Input</h3>
          <MatrixDisplay
            data={inputs}
            label="Attention Output"
            rowLabels={labels}
            columnLabels={modelDimLabels}
            maxAbsValue={0.2}
            cellSize="sm"
            selectable={false}
            matrixType="none"
          />
        </div>

        {/* Second Column: Weights and biases */}
        {showSteps && (
          <div className="col-span-3">
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700">Weights</h3>
            <div className="grid grid-cols-1 gap-1">
              <div>
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">W₁</h4>
                <MatrixDisplay
                  data={W1}
                  rowLabels={modelDimLabels}
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.1}
                  cellSize="sm"
                  selectable={false}
                  matrixType="none"
                />
              </div>
              <div>
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">W₂</h4>
                <MatrixDisplay
                  data={W2}
                  rowLabels={ffnDimLabels}
                  columnLabels={modelDimLabels}
                  maxAbsValue={0.1}
                  cellSize="sm"
                  selectable={false}
                  matrixType="none"
                />
              </div>
            </div>
          </div>
        )}

        {/* Third Column: Intermediate results */}
        {showSteps && (
          <div className="col-span-3">
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700">Steps</h3>
            <div className="grid grid-cols-2 gap-1">
              <div>
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">xW₁+b₁</h4>
                <MatrixDisplay
                  data={firstLayerOutput}
                  rowLabels={labels}
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.5}
                  cellSize="sm"
                  selectable={false}
                  matrixType="none"
                />
              </div>
              <div>
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700">ReLU</h4>
                <MatrixDisplay
                  data={activations}
                  rowLabels={labels}
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.5}
                  cellSize="sm"
                  selectable={false}
                  matrixType="none"
                />
              </div>
            </div>
          </div>
        )}

        {/* Right Column: Output */}
        <div className="col-span-3">
          <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700">Output</h3>
          <MatrixDisplay
            data={output}
            label="FFN(x)"
            rowLabels={labels}
            columnLabels={modelDimLabels}
            maxAbsValue={0.2}
            cellSize="sm"
            selectable={false}
            matrixType="none"
          />
        </div>
      </div>
    </div>
  );
};

export default FeedForward;