import React, { useState, useEffect, useCallback, useMemo } from 'react';
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

  // Token labels for 6 tokens - a simple sentence
  const tokenLabels = ["The", "cat", "sat", "on", "the", "mat"];

  // Sample data generation
  const [embeddings, setEmbeddings] = useState(() =>
    generateSampleEmbeddings(numTokens, embeddingDim)
  );

  const [attentionWeights, setAttentionWeights] = useState(() =>
    generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
  );

  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights] = useState(() =>
    generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
  );

  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

  // State for the currently selected element - includes matrix type, row and column
  const [selectedElement, setSelectedElement] = useState<{
    matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'none';
    row: number;
    col: number;
  } | null>(null);

  // State for the current value of the selected element
  const [selectedValue, setSelectedValue] = useState<number | null>(null);

  // Update selectedValue when selectedElement changes
  useEffect(() => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        setSelectedValue(embeddings[row][col]);
      } else if (matrixType === 'weightQ') {
        setSelectedValue(attentionWeights.weightQ[row][col]);
      } else if (matrixType === 'weightK') {
        setSelectedValue(attentionWeights.weightK[row][col]);
      } else if (matrixType === 'weightV') {
        setSelectedValue(attentionWeights.weightV[row][col]);
      }
    } else {
      setSelectedValue(null);
    }
  }, [selectedElement, embeddings, attentionWeights]);

  // Handle element selection in matrices
  const handleElementClick = useCallback((
    matrixType: 'embeddings' | 'weightQ' | 'weightK' | 'weightV' | 'none',
    row: number,
    col: number
  ) => {
    // Toggle selection if clicking the same element
    if (selectedElement &&
        selectedElement.matrixType === matrixType &&
        selectedElement.row === row &&
        selectedElement.col === col) {
      setSelectedElement(null);
    } else {
      setSelectedElement({ matrixType, row, col });
    }
  }, [selectedElement]);

  // Create value label for the selected element
  const valueLabel = useMemo(() => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        return `${tokenLabels[row]}.d${col+1}`;
      } else if (matrixType === 'weightQ') {
        return `W^Q[${row+1},${col+1}]`;
      } else if (matrixType === 'weightK') {
        return `W^K[${row+1},${col+1}]`;
      } else if (matrixType === 'weightV') {
        return `W^V[${row+1},${col+1}]`;
      }
    }
    return undefined;
  }, [selectedElement, tokenLabels]);

  // Define a constant for max absolute value of embeddings
  const maxAbsValue = 0.5;

  // Handle value change from the slider
  const handleValueChange = useCallback((newValue: number) => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        // Create a new copy of embeddings with the updated value
        const newEmbeddings = embeddings.map((r, i) =>
          i === row
            ? r.map((v, j) => j === col ? newValue : v)
            : [...r]
        );
        setEmbeddings(newEmbeddings);
      }
      else if (matrixType === 'weightQ' || matrixType === 'weightK' || matrixType === 'weightV') {
        // Create a deep copy of attention weights
        const newAttentionWeights = {
          weightQ: [...attentionWeights.weightQ.map(r => [...r])],
          weightK: [...attentionWeights.weightK.map(r => [...r])],
          weightV: [...attentionWeights.weightV.map(r => [...r])]
        };

        // Update the specific weight
        if (matrixType === 'weightQ') {
          newAttentionWeights.weightQ[row][col] = newValue;
        } else if (matrixType === 'weightK') {
          newAttentionWeights.weightK[row][col] = newValue;
        } else if (matrixType === 'weightV') {
          newAttentionWeights.weightV[row][col] = newValue;
        }

        setAttentionWeights(newAttentionWeights);
      }

      setSelectedValue(newValue);
    }
  }, [selectedElement, embeddings, attentionWeights]);

  
  // Handler for receiving the computed context from the attention head
  const handleAttentionContextComputed = (context: number[][]) => {
    setAttentionContext(context);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="w-full p-0.5">
        <div className="bg-white rounded p-0.5 mb-0.5">

          <div className="mb-0.5">
            <h3 className="text-sm font-semibold mb-0.5 border-b pb-0.5">
              Self-Attention
            </h3>
            {/* Value Adjuster will be shown in the EmbeddingElement component */}

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
              onValueChange={handleValueChange}
            />
          </div>
          
          <div className="mt-0.5">
            <h3 className="text-sm font-semibold mb-0.5 border-b pb-0.5">
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
              <div className="p-0.5 bg-gray-100 rounded">
                <p className="text-gray-600 italic text-[0.6rem]">
                  Waiting for attention computation...
                </p>
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded p-0.5 text-[0.6rem]">
          <p className="text-gray-700">
            Blue: positive, Pink: negative. Click a value to edit (magenta border).
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;