import React, { useMemo } from 'react';
import MatrixDisplay from './MatrixDisplay';
import { 
  matrixMultiply, 
  addBias, 
  applyFn, 
  relu, 
  isPortraitOrientation 
} from '../utils/matrixOperations';

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
  // Activation function to use (defaults to ReLU)
  activationFn?: (x: number) => number;
  // Name of the activation function for display
  activationFnName?: string;
  // Label for value editing
  valueLabel?: string;
  // Optional callback when output is computed
  onOutputComputed?: (output: number[][]) => void;
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
  showSteps = true,
  selectedElement = null,
  onElementClick,
  onValueChange,
  activationFn = relu,
  activationFnName = 'ReLU',
  valueLabel,
  onOutputComputed
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
    Array.from({ length: d_ff }, (_, i) => ``), // Empty strings for hidden dimensions
    [d_ff]
  );
  
  const labels = tokenLabels || defaultTokenLabels;

  // First layer computation: inputs × W1 + b1
  const firstLayerOutput = useMemo(() => {
    const product = matrixMultiply(inputs, W1);
    return addBias(product, b1);
  }, [inputs, W1, b1]);

  // Apply activation function (defaults to ReLU)
  const activations = useMemo(() => 
    applyFn(firstLayerOutput, activationFn),
    [firstLayerOutput, activationFn]
  );
  
  // Use activations directly (no dropout)
  const activationsWithDropout = activations;

  // Second layer computation: activations × W2 + b2
  const output = useMemo(() => {
    const product = matrixMultiply(activationsWithDropout, W2);
    return addBias(product, b2);
  }, [activationsWithDropout, W2, b2]);
  
  // Call the output computed callback when the output changes
  React.useEffect(() => {
    if (onOutputComputed && output.length > 0) {
      onOutputComputed(output);
    }
  }, [output, onOutputComputed]);

  return (
    <div className="flex flex-col gap-0.5 p-0.5 bg-white rounded">
      {/* Main Container - Responsive layout based on orientation */}
      <div className={`flex flex-col ${!isPortraitOrientation() ? 'md:grid md:grid-cols-3' : ''} lg:grid lg:grid-cols-9 gap-3 lg:gap-1`}>
        {/* Weights and biases - Full width in portrait, 1/3 in landscape on mobile, 3/9 on desktop */}
        {showSteps && (
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-3 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Weights</h3>
            <div className={`${!isPortraitOrientation() ? 'grid grid-cols-1' : 'grid grid-cols-2'} gap-1 w-full`}>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">W₁</h4>
                <MatrixDisplay
                  data={W1}
                  rowLabels={modelDimLabels}
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.1}
                  cellSize="xs" // Using xs size for hidden layer connections
                  selectable={true}
                  selectedElement={selectedElement?.matrixType === 'weightW1' ? selectedElement : null}
                  matrixType="weightW1"
                  onElementClick={onElementClick}
                  onValueChange={onValueChange}
                  valueLabel={selectedElement?.matrixType === 'weightW1' ? valueLabel : undefined}
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">W₂</h4>
                <MatrixDisplay
                  data={W2}
                  rowLabels={ffnDimLabels}
                  columnLabels={modelDimLabels}
                  maxAbsValue={0.1}
                  cellSize="xs" // Using xs size for hidden layer connections
                  selectable={true}
                  selectedElement={selectedElement?.matrixType === 'weightW2' ? selectedElement : null}
                  matrixType="weightW2"
                  onElementClick={onElementClick}
                  onValueChange={onValueChange}
                  valueLabel={selectedElement?.matrixType === 'weightW2' ? valueLabel : undefined}
                />
              </div>
            </div>
          </div>
        )}

        {/* Intermediate results - Full width in portrait, 1/3 in landscape on mobile, 3/9 on desktop */}
        {showSteps && (
          <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-3 flex flex-col items-center justify-center`}>
            <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Steps</h3>
            <div className={`${!isPortraitOrientation() ? 'grid grid-cols-1' : 'grid grid-cols-2'} gap-1 w-full`}>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">xW₁+b₁</h4>
                <MatrixDisplay
                  data={firstLayerOutput}
                  rowLabels={[]} // Removed token labels as they don't make sense for hidden layer
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.5}
                  cellSize="xs" // Using xs size for hidden layer
                  selectable={false}
                  matrixType="none"
                />
              </div>
              <div className="flex flex-col items-center justify-center">
                <h4 className="text-[0.5rem] font-medium mb-0.5 text-center text-gray-700 w-full">
                  {activationFnName}
                </h4>
                <MatrixDisplay
                  data={activations}
                  rowLabels={[]} // Removed token labels as they don't make sense for hidden layer
                  columnLabels={ffnDimLabels}
                  maxAbsValue={0.5}
                  cellSize="xs" // Using xs size for hidden layer
                  selectable={false}
                  matrixType="none"
                />
              </div>
            </div>
          </div>
        )}

        {/* Output - Full width in portrait, 1/3 in landscape on mobile, 3/9 on desktop */}
        <div className={`${!isPortraitOrientation() ? 'md:col-span-1' : ''} lg:col-span-3 flex flex-col items-center justify-center`}>
          <h3 className="text-[0.65rem] font-semibold mb-0.5 text-gray-700 text-center w-full">Output</h3>
          <MatrixDisplay
            data={output}
            label="FFN(x)"
            rowLabels={labels}
            columnLabels={modelDimLabels}
            maxAbsValue={0.2}
            cellSize="xs"
            selectable={false}
            matrixType="none"
          />
        </div>
      </div>
    </div>
  );
};

export default FeedForward;