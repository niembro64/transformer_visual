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
      <main className="w-full p-1">
        <div className="bg-white rounded p-1 mb-1">

          <div className="mb-1">
            <h3 className="text-base font-semibold mb-1 border-b pb-1">
              Self-Attention
            </h3>
            
            {/* Value Adjuster Slider */}
            {selectedElement !== null && selectedValue !== null && (
              <div className="mb-1 mt-1 p-1 bg-yellow-50 border border-yellow-200 rounded text-xs">
                <div className="flex items-center gap-1">
                  <span className="whitespace-nowrap">{tokenLabels[selectedElement[0]]}.{selectedElement[1] + 1}:</span>
                  <input
                    type="range"
                    min={-maxAbsValue}
                    max={maxAbsValue}
                    step={maxAbsValue / 50}
                    value={selectedValue}
                    onChange={(e) => handleValueChange(parseFloat(e.target.value))}
                    className="flex-grow h-3"
                  />
                  <span className="font-mono text-2xs w-12">{selectedValue.toExponential(2)}</span>
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
          
          <div className="mt-1">
            <h3 className="text-base font-semibold mb-1 border-b pb-1">
              Feed-Forward Network
            </h3>

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
              <div className="p-1 bg-gray-100 rounded">
                <p className="text-gray-600 italic text-xs">
                  Waiting for attention computation...
                </p>
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded p-1 text-xs">
          <p className="text-gray-700">
            Blue: positive, Pink: negative. Click a value to edit (magenta border).
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;