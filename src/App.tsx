import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import {
  generateSampleEmbeddings,
  generateSampleAttentionWeights,
  generateSampleMLPWeights
} from './utils/matrixOperations';

function App() {
  // Configuration for the demo with new dimensions
  const numTokens = 6;        // Number of tokens in the sequence (6 tokens)
  const embeddingDim = 4;     // Dimension of token embeddings (d_model = 4)
  const attentionHeadDim = 3; // Dimension of attention head (d_k/d_v = 3)
  const mlpHiddenDim = 8;     // Dimension of MLP hidden layer (d_ff = 8, typically 4x d_model)

  // Sample data generation
  const [embeddings, setEmbeddings] = useState(() =>
    generateSampleEmbeddings(numTokens, embeddingDim)
  );

  const [attentionWeights] = useState(() =>
    generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
  );

  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights] = useState(() =>
    generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
  );

  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

  // State for the currently selected element in the embeddings matrix
  const [selectedElement, setSelectedElement] = useState<[number, number] | null>(null);

  // State for the current value of the selected element
  const [selectedValue, setSelectedValue] = useState<number | null>(null);

  // Update selectedValue when selectedElement changes
  useEffect(() => {
    if (selectedElement) {
      const [row, col] = selectedElement;
      setSelectedValue(embeddings[row][col]);
    } else {
      setSelectedValue(null);
    }
  }, [selectedElement, embeddings]);

  // Handle element selection in the embeddings matrix
  const handleElementClick = useCallback((row: number, col: number) => {
    // Toggle selection if clicking the same element
    if (selectedElement && selectedElement[0] === row && selectedElement[1] === col) {
      setSelectedElement(null);
    } else {
      setSelectedElement([row, col]);
    }
  }, [selectedElement]);

  // Define a constant for max absolute value of embeddings
  const maxAbsValue = 0.5;

  // Handle value change from the slider
  const handleValueChange = useCallback((newValue: number) => {
    if (selectedElement) {
      const [row, col] = selectedElement;
      // Create a new copy of embeddings with the updated value
      const newEmbeddings = embeddings.map((r, i) =>
        i === row
          ? r.map((v, j) => j === col ? newValue : v)
          : [...r]
      );
      setEmbeddings(newEmbeddings);
      setSelectedValue(newValue);
    }
  }, [selectedElement, embeddings]);

  // Token labels for 6 tokens - a simple sentence
  const tokenLabels = ["The", "cat", "sat", "on", "the", "mat"];
  
  // Handler for receiving the computed context from the attention head
  const handleAttentionContextComputed = (context: number[][]) => {
    setAttentionContext(context);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="w-full p-2">
        <div className="bg-white rounded p-3 mb-3">
          <h2 className="text-xl font-bold mb-2">Transformer Components</h2>
          
          <div className="mb-4">
            <h3 className="text-lg font-semibold mb-2 border-b pb-1">
              Part 1: Self-Attention Mechanism
            </h3>
            <p className="text-gray-700 mb-2 text-sm">
              The self-attention mechanism allows each token to gather information from all other tokens
              in the sequence, weighting their relevance.
            </p>
            
            {/* Value Adjuster Slider */}
            {selectedElement !== null && selectedValue !== null && (
              <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded">
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-semibold">
                    Editing {tokenLabels[selectedElement[0]]}.{selectedElement[1] + 1}:
                  </span>
                  <input
                    type="range"
                    min={-maxAbsValue}
                    max={maxAbsValue}
                    step={maxAbsValue / 50}
                    value={selectedValue}
                    onChange={(e) => handleValueChange(parseFloat(e.target.value))}
                    className="flex-grow h-4"
                  />
                  <span className="font-mono">{selectedValue.toExponential(2)}</span>
                </div>
              </div>
            )}

            <AttentionHead
              embeddings={embeddings}
              weightQ={attentionWeights.weightQ}
              weightK={attentionWeights.weightK}
              weightV={attentionWeights.weightV}
              tokenLabels={tokenLabels}
              showSteps={true}
              onContextComputed={handleAttentionContextComputed}
              selectedElement={selectedElement}
              onElementClick={handleElementClick}
            />
          </div>
          
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2 border-b pb-1">
              Part 2: Position-wise Feed-Forward Network
            </h3>
            <p className="text-gray-700 mb-2 text-sm">
              After the attention mechanism, each token passes through a
              feed-forward neural network applied independently to each position.
            </p>
            
            {attentionContext.length > 0 ? (
              <FeedForward
                inputs={attentionContext} // Using the output from the attention layer
                W1={mlpWeights.W1}
                b1={mlpWeights.b1}
                W2={mlpWeights.W2}
                b2={mlpWeights.b2}
                tokenLabels={tokenLabels}
                showSteps={true}
              />
            ) : (
              <div className="p-2 bg-gray-100 rounded">
                <p className="text-gray-600 italic text-xs">
                  Feed-Forward Network will appear after attention computation is complete.
                </p>
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded p-2 text-sm">
          <h2 className="text-base font-bold mb-1">About This Visualization</h2>
          <p className="text-gray-700 mb-1">
            This visualization demonstrates a transformer with simplified dimensions: {embeddingDim} (d_model),
            {attentionHeadDim} (d_k/d_v), {mlpHiddenDim} (d_ff).
          </p>
          <div className="text-gray-700 mb-1 grid grid-cols-2">
            <p>• Token Sequence: {tokenLabels.join(", ")}</p>
            <p>• Pale blue: positive, Pale pink: negative, Light gray: near-zero</p>
          </div>
          <p className="text-gray-700 text-xs">
            Click on any value in the top embedding matrix to edit it with the slider. All downstream calculations will update automatically.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;