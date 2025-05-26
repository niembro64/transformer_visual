import React, { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import MatrixDisplay from './components/MatrixDisplay';
import {
  addPositionalEncodings,
  applyDropout,
  applyRandomWalk,
  applyRandomWalkToVector,
  generatePositionalEncodings,
  generateSampleAttentionWeights,
  generateSampleEmbeddings,
  generateSampleMLPWeights,
  isPortraitOrientation,
  relu,
  vectorDotProduct,
  mseLoss,
  mseGradient,
  updateWeights,
  updateVectorWeights,
} from './utils/matrixOperations';

export const dropoutUniveral = 0.03;

function App() {
  // Fixed dimension values
  const embeddingDim = isPortraitOrientation() ? 4 : 8; // Dimension of embeddings (d_model)
  const attentionHeadDim = isPortraitOrientation() ? 2 : 4; // Dimension of attention heads (d_k = d_v = d_model / num_heads)
  const mlpHiddenDim = 4; // Dimension of MLP hidden layer (d_ff = 8, typically 4x d_model)

  // Dropout rates
  const embeddingDropoutRate = dropoutUniveral; // Dropout rate after embeddings + positional encodings
  const attentionDropoutRate = dropoutUniveral; // Dropout rate after attention
  const ffnDropoutRate = dropoutUniveral; // Dropout rate in feed-forward network

  // Training mode - determines if dropout is applied and weights are updated
  const [trainingMode, setTrainingMode] = useState(false);
  // Target output token for training (what we're trying to predict)
  const [targetTokenIndex, setTargetTokenIndex] = useState<number | null>(null);

  // Vocabulary of 25 common words
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const vocabularyWords: string[] = [
    'the', // 0
    'a', // 1
    'and', // 2
    'to', // 3
    'is', // 4
    'in', // 5
    'it', // 6
    'of', // 7
    'that', // 8
    'boy', // 9
  ];

  // Generate vocabulary embeddings - mutable state
  const [vocabularyEmbeddings, setVocabularyEmbeddings] = useState(() =>
    generateSampleEmbeddings(vocabularyWords.length, embeddingDim)
  );

  // Track selected tokens (indices into vocabulary)
  const [selectedTokenIndices, setSelectedTokenIndices] = useState<number[]>([
    1, // a
    9, // boy
    5, // with
  ]);

  // Get token labels from selected indices
  const tokenLabels = useMemo(
    () => selectedTokenIndices.map((idx) => vocabularyWords[idx]),
    [selectedTokenIndices, vocabularyWords]
  );

  // Maximum sequence length - based on current token count
  const maxSeqLength = useMemo(
    () => selectedTokenIndices.length * 2,
    [selectedTokenIndices.length]
  ); // Allow room for growth

  // Generate positional encodings - regenerate when embedding dimension changes
  const [positionalEncodings, setPositionalEncodings] = useState(() =>
    generatePositionalEncodings(maxSeqLength, embeddingDim)
  );

  // Update positional encodings when dimensions change
  useEffect(() => {
    setPositionalEncodings(
      generatePositionalEncodings(maxSeqLength, embeddingDim)
    );
  }, [maxSeqLength, embeddingDim]);

  // Get embeddings for selected tokens
  const rawEmbeddings = useMemo(() => {
    return selectedTokenIndices.map((idx) => [...vocabularyEmbeddings[idx]]);
  }, [selectedTokenIndices, vocabularyEmbeddings]);

  // Update vocabulary embeddings when dimension changes
  useEffect(() => {
    setVocabularyEmbeddings(
      generateSampleEmbeddings(vocabularyWords.length, embeddingDim)
    );
  }, [embeddingDim]);

  // Apply positional encodings to embeddings
  const embeddings = useMemo(
    () => addPositionalEncodings(rawEmbeddings, positionalEncodings),
    [rawEmbeddings, positionalEncodings]
  );

  // For forcing re-renders on dropout timer cycles and weight updates
  const [trainingCycle, setTrainingCycle] = useState(0);

  // Track if device is in mobile mode (height > width)
  const [isMobile, setIsMobile] = useState(isPortraitOrientation);

  // Listen for orientation/size changes
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(isPortraitOrientation());
    };

    // Set initial value
    handleResize();

    // Add event listener
    window.addEventListener('resize', handleResize);

    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Used for random walk step size - smaller values for more subtle updates
  const weightUpdateStepSize = 0; // Set to 0 to disable random walk, use pure gradient descent
  
  // Learning rate for gradient descent training
  const learningRate = 0.1; // Moderate learning rate for cross-entropy loss
  
  // Track training loss
  const [trainingLoss, setTrainingLoss] = useState<number | null>(null);
  
  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

  // State to hold the feed-forward network output (final layer prediction)
  const [ffnOutput, setFfnOutput] = useState<number[][]>([]);
  
  // Type for selected element
  type ElementObject = {
    matrixType:
      | 'embeddings'
      | 'weightQ'
      | 'weightK'
      | 'weightV'
      | 'weightW1'
      | 'weightW2'
      | 'none';
    row: number;
    col: number;
  };
  
  // Only have one selectedElement for the entire application to prevent multiple sliders
  const [selectedElement, setSelectedElement] = useState<ElementObject | null>(
    null
  );

  // Timer to update dropout masks and perform weight updates every second in training mode
  useEffect(() => {
    let timerId: number | null = null;

    if (trainingMode && targetTokenIndex !== null && ffnOutput.length > 0) {
      // Start a timer that updates every second
      timerId = window.setInterval(() => {
        setTrainingCycle((prev) => prev + 1); // Increment counter to trigger re-renders

        // Perform gradient-based weight updates when we have a target
        if (trainingMode && targetTokenIndex !== null && ffnOutput.length > 0) {
          // Get the predicted embedding (last token's output)
          const predictedEmbedding = ffnOutput[ffnOutput.length - 1];
          
          // Calculate which token is predicted by finding closest vocabulary embedding
          const dotProducts = vocabularyEmbeddings.map((vocabEmbedding) =>
            vectorDotProduct(predictedEmbedding, vocabEmbedding)
          );
          
          // Apply softmax to get probabilities
          const maxDotProduct = Math.max(...dotProducts);
          const expValues = dotProducts.map((dp) => Math.exp(dp - maxDotProduct));
          const sumExp = expValues.reduce((a, b) => a + b, 0);
          const probabilities = expValues.map((exp) => exp / sumExp);
          
          // Cross-entropy loss: -log(probability of correct token)
          const targetProb = probabilities[targetTokenIndex];
          const loss = -Math.log(Math.max(targetProb, 1e-7)); // Avoid log(0)
          setTrainingLoss(loss);
          
          // Gradient for cross-entropy with softmax
          // For the correct class: gradient = predicted_prob - 1
          // For other classes: gradient = predicted_prob
          const gradients = probabilities.map((prob, idx) => 
            idx === targetTokenIndex ? prob - 1 : prob
          );
          
          // Convert gradient to embedding space
          // We need to push the output embedding towards the target embedding
          const targetEmbedding = vocabularyEmbeddings[targetTokenIndex];
          const outputGradient = predictedEmbedding.map((val, i) => 
            val - targetEmbedding[i]
          );
          
          // Debug logging
          const predictedTokenIndex = dotProducts.indexOf(Math.max(...dotProducts));
          console.log('Training update:', {
            loss: loss.toFixed(4),
            targetToken: vocabularyWords[targetTokenIndex],
            predictedToken: vocabularyWords[predictedTokenIndex],
            targetProb: targetProb.toFixed(4),
            learningRate
          });
          
          // Apply gradient-based updates to all layers
          // Simplified backpropagation - in a real implementation we'd compute proper gradients
          
          // Update MLP weights with gradient descent
          setMlpWeights((prev) => {
            const newWeights = {
              W1: [...prev.W1.map((r) => [...r])],
              b1: [...prev.b1],
              W2: [...prev.W2.map((r) => [...r])],
              b2: [...prev.b2],
            };
            
            // Apply gradient updates to W2 (output layer)
            // The gradient for W2 is approximately: gradient = output_gradient * hidden_activation^T
            // Skip selected element if it's in W2
            if (selectedElement?.matrixType === 'weightW2') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.W2.length; i++) {
                for (let j = 0; j < newWeights.W2[i].length; j++) {
                  if (i !== row || j !== col) {
                    // Simplified gradient - in reality would use hidden layer activations
                    const gradient = outputGradient[i] * 1.0; // Increased from 0.1
                    newWeights.W2[i][j] -= learningRate * gradient;
                  }
                }
              }
            } else {
              // Apply gradient descent
              for (let i = 0; i < newWeights.W2.length; i++) {
                for (let j = 0; j < newWeights.W2[i].length; j++) {
                  // Simplified gradient - in reality would use hidden layer activations
                  const gradient = outputGradient[i] * 1.0; // Increased from 0.1
                  newWeights.W2[i][j] -= learningRate * gradient;
                }
              }
            }
            
            // Apply gradient updates to biases
            for (let i = 0; i < newWeights.b2.length; i++) {
              newWeights.b2[i] -= learningRate * outputGradient[i] * 1.0; // Increased from 0.1
            }
            
            // For W1 and b1, we'd need to backpropagate through activation function
            // For now, apply small gradient updates proportional to output error
            if (selectedElement?.matrixType === 'weightW1') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.W1.length; i++) {
                for (let j = 0; j < newWeights.W1[i].length; j++) {
                  if (i !== row || j !== col) {
                    // Simplified gradient
                    const avgError = outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;
                    newWeights.W1[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.W1.length; i++) {
                for (let j = 0; j < newWeights.W1[i].length; j++) {
                  // Simplified gradient
                  const avgError = outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;
                  newWeights.W1[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                }
              }
            }
            
            // Update b1 biases
            const avgError = outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;
            for (let i = 0; i < newWeights.b1.length; i++) {
              newWeights.b1[i] -= learningRate * avgError * 0.5; // Increased from 0.01
            }
            
            return newWeights;
          });
          
          // Update attention weights with gradient descent
          setAttentionWeights((prev) => {
            const newWeights = {
              weightQ: [...prev.weightQ.map((r) => [...r])],
              weightK: [...prev.weightK.map((r) => [...r])],
              weightV: [...prev.weightV.map((r) => [...r])],
            };
            
            // For attention weights, the gradient flows back from the output through the attention mechanism
            // This is a simplified version - proper gradients would require full backpropagation
            const avgError = outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;
            
            // Update Q matrix with gradient descent (skip selected element)
            if (selectedElement?.matrixType === 'weightQ') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightQ.length; i++) {
                for (let j = 0; j < newWeights.weightQ[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightQ[i][j] -= learningRate * avgError * 0.1; // Increased from 0.001
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightQ.length; i++) {
                for (let j = 0; j < newWeights.weightQ[i].length; j++) {
                  newWeights.weightQ[i][j] -= learningRate * avgError * 0.1; // Increased from 0.001
                }
              }
            }
            
            // Update K matrix with gradient descent (skip selected element)
            if (selectedElement?.matrixType === 'weightK') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightK.length; i++) {
                for (let j = 0; j < newWeights.weightK[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightK[i][j] -= learningRate * avgError * 0.1; // Increased from 0.001
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightK.length; i++) {
                for (let j = 0; j < newWeights.weightK[i].length; j++) {
                  newWeights.weightK[i][j] -= learningRate * avgError * 0.1; // Increased from 0.001
                }
              }
            }
            
            // Update V matrix with gradient descent - larger updates as it affects output directly
            if (selectedElement?.matrixType === 'weightV') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightV.length; i++) {
                for (let j = 0; j < newWeights.weightV[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightV[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightV.length; i++) {
                for (let j = 0; j < newWeights.weightV[i].length; j++) {
                  newWeights.weightV[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                }
              }
            }
            
            return newWeights;
          });
          
          // Update vocabulary embeddings with gradient descent
          setVocabularyEmbeddings((prev) => {
            const newEmbeddings = prev.map((emb, idx) => {
              if (idx === targetTokenIndex) {
                // Move target embedding to reduce loss
                // The gradient for target embedding is negative of output gradient
                return emb.map((val, i) => 
                  val - learningRate * outputGradient[i] * 1.0 // Increased from 0.1
                );
              }
              // Other embeddings don't get updated in this simplified version
              // In a real implementation, all embeddings that contributed to the output would be updated
              return [...emb];
            });
            return newEmbeddings;
          });
        }
      }, 1000);
    } else if (trainingMode && targetTokenIndex === null) {
      // No updates when no target is selected - pure gradient descent needs a target
      // Just update the training cycle for dropout mask updates
      timerId = window.setInterval(() => {
        setTrainingCycle((prev) => prev + 1); // Increment counter to trigger re-renders for dropout
      }, 1000);
    }

    return () => {
      if (timerId !== null) {
        window.clearInterval(timerId);
      }
    };
  }, [trainingMode, weightUpdateStepSize, targetTokenIndex, ffnOutput, vocabularyEmbeddings, learningRate, selectedElement]);

  // Apply dropout to embeddings (only during training) with a unique ID
  // Include trainingCycle in dependencies to ensure recalculation when timer ticks
  const embeddingsWithDropout = useMemo(
    () =>
      applyDropout(
        embeddings,
        embeddingDropoutRate,
        trainingMode,
        'embeddings_dropout'
      ),
    [embeddings, embeddingDropoutRate, trainingMode, trainingCycle]
  );

  // Sample data generation for attention weights - regenerate when dimensions change
  const [attentionWeights, setAttentionWeights] = useState(() =>
    generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
  );

  // Update attention weights when dimensions change
  useEffect(() => {
    setAttentionWeights(
      generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
    );
  }, [embeddingDim, attentionHeadDim]);

  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights, setMlpWeights] = useState(() =>
    generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
  );

  // Update MLP weights when dimensions change
  useEffect(() => {
    setMlpWeights(
      generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
    );
  }, [embeddingDim, mlpHiddenDim, attentionHeadDim]);

  // No need for these functions anymore - handled by drag and drop

  // Click interaction state
  const [recentlyAddedIndex, setRecentlyAddedIndex] = useState<number | null>(
    null
  );

  // Handler to add a token from tokenizer to input sequence
  const handleTokenizerClick = useCallback((index: number) => {
    setSelectedTokenIndices((prev) => [...prev, index]);
    setRecentlyAddedIndex(index);

    // Clear the highlight after a brief delay
    setTimeout(() => {
      setRecentlyAddedIndex(null);
    }, 500);
  }, []);

  // Handler to remove a token from the input sequence
  const handleSequenceTokenClick = useCallback((index: number) => {
    setSelectedTokenIndices((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // No need for dimension adjustment functions anymore

  const initialElement: ElementObject | null = useMemo(() => {
    // Initialize selectedElement to the second-to-last token's 4th embedding
    if (selectedTokenIndices.length > 0 && embeddingDim > 0) {
      const tokenIndex = Math.max(0, selectedTokenIndices.length - 2); // Second-to-last token
      const embIndex = Math.min(3, embeddingDim - 1); // 4th embedding (index 3) or last if fewer

      return {
        matrixType: 'embeddings',
        row: tokenIndex,
        col: embIndex,
      };
    }
    return null;
  }, [selectedTokenIndices, embeddingDim]);

  // State for the current value of the selected element
  const [selectedValue, setSelectedValue] = useState<number | null>(null);

  // Update selectedValue when selectedElement changes
  useEffect(() => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        // Check if row is within current selection
        if (row < rawEmbeddings.length && col < rawEmbeddings[0].length) {
          setSelectedValue(rawEmbeddings[row][col]); // Show original embedding values before positional encoding
        } else {
          setSelectedValue(null);
        }
      } else if (matrixType === 'weightQ') {
        setSelectedValue(attentionWeights.weightQ[row][col]);
      } else if (matrixType === 'weightK') {
        setSelectedValue(attentionWeights.weightK[row][col]);
      } else if (matrixType === 'weightV') {
        setSelectedValue(attentionWeights.weightV[row][col]);
      } else if (matrixType === 'weightW1') {
        setSelectedValue(mlpWeights.W1[row][col]);
      } else if (matrixType === 'weightW2') {
        setSelectedValue(mlpWeights.W2[row][col]);
      }
    } else {
      setSelectedValue(null);
    }
  }, [selectedElement, rawEmbeddings, attentionWeights, mlpWeights]);

  // Handle element selection in matrices
  const handleElementClick = useCallback(
    (
      matrixType:
        | 'embeddings'
        | 'weightQ'
        | 'weightK'
        | 'weightV'
        | 'weightW1'
        | 'weightW2'
        | 'none',
      row: number,
      col: number
    ) => {
      // Toggle selection if clicking the same element
      if (
        selectedElement &&
        selectedElement.matrixType === matrixType &&
        selectedElement.row === row &&
        selectedElement.col === col
      ) {
        setSelectedElement(null);
      } else {
        setSelectedElement({ matrixType, row, col });
      }
    },
    [selectedElement]
  );

  // Create value label for the selected element
  const valueLabel = useMemo(() => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        return `${tokenLabels[row]}.d${col + 1}`;
      } else if (matrixType === 'weightQ') {
        return `W^Q[${row + 1},${col + 1}]`;
      } else if (matrixType === 'weightK') {
        return `W^K[${row + 1},${col + 1}]`;
      } else if (matrixType === 'weightV') {
        return `W^V[${row + 1},${col + 1}]`;
      } else if (matrixType === 'weightW1') {
        return `W₁[${row + 1},${col + 1}]`;
      } else if (matrixType === 'weightW2') {
        return `W₂[${row + 1},${col + 1}]`;
      }
    }
    return undefined;
  }, [selectedElement, tokenLabels]);

  // Handle value change from the slider
  const handleValueChange = useCallback(
    (newValue: number) => {
      if (selectedElement) {
        const { matrixType, row, col } = selectedElement;

        if (matrixType === 'embeddings') {
          // Update the vocabulary embedding for the specific token
          if (row < selectedTokenIndices.length && col < embeddingDim) {
            const tokenVocabIdx = selectedTokenIndices[row];
            // Create a new copy of vocabulary embeddings with the updated value
            const newVocabEmbeddings = vocabularyEmbeddings.map((r, i) =>
              i === tokenVocabIdx
                ? r.map((v, j) => (j === col ? newValue : v))
                : [...r]
            );
            setVocabularyEmbeddings(newVocabEmbeddings);
          }
        } else if (
          matrixType === 'weightQ' ||
          matrixType === 'weightK' ||
          matrixType === 'weightV'
        ) {
          // Create a deep copy of attention weights
          const newAttentionWeights = {
            weightQ: [...attentionWeights.weightQ.map((r) => [...r])],
            weightK: [...attentionWeights.weightK.map((r) => [...r])],
            weightV: [...attentionWeights.weightV.map((r) => [...r])],
          };

          // Update the specific weight
          if (
            matrixType === 'weightQ' &&
            row < attentionWeights.weightQ.length &&
            col < attentionWeights.weightQ[0].length
          ) {
            newAttentionWeights.weightQ[row][col] = newValue;
            setAttentionWeights(newAttentionWeights);
          } else if (
            matrixType === 'weightK' &&
            row < attentionWeights.weightK.length &&
            col < attentionWeights.weightK[0].length
          ) {
            newAttentionWeights.weightK[row][col] = newValue;
            setAttentionWeights(newAttentionWeights);
          } else if (
            matrixType === 'weightV' &&
            row < attentionWeights.weightV.length &&
            col < attentionWeights.weightV[0].length
          ) {
            newAttentionWeights.weightV[row][col] = newValue;
            setAttentionWeights(newAttentionWeights);
          }
        } else if (matrixType === 'weightW1' || matrixType === 'weightW2') {
          // Create a deep copy of MLP weights
          const newMlpWeights = {
            W1: [...mlpWeights.W1.map((r) => [...r])],
            b1: [...mlpWeights.b1],
            W2: [...mlpWeights.W2.map((r) => [...r])],
            b2: [...mlpWeights.b2],
          };

          // Update the specific weight
          if (
            matrixType === 'weightW1' &&
            row < mlpWeights.W1.length &&
            col < mlpWeights.W1[0].length
          ) {
            newMlpWeights.W1[row][col] = newValue;
            setMlpWeights(newMlpWeights);
          } else if (
            matrixType === 'weightW2' &&
            row < mlpWeights.W2.length &&
            col < mlpWeights.W2[0].length
          ) {
            newMlpWeights.W2[row][col] = newValue;
            setMlpWeights(newMlpWeights);
          }
        }

        setSelectedValue(newValue);
      }
    },
    [selectedElement, rawEmbeddings, attentionWeights, mlpWeights]
  );

  // Handler for receiving the computed context from the attention head
  const handleAttentionContextComputed = (context: number[][]) => {
    setAttentionContext(context);
  };

  // Handler for receiving the computed output from the feed-forward network
  const handleFfnOutputComputed = (output: number[][]) => {
    setFfnOutput(output);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="w-full p-0.5 md:p-2">
        <div className="bg-white rounded p-0.5 mb-0.5">
          {/* Main control panel */}
          <div className="mb-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-sm overflow-hidden">
            {/* Header bar */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-2 sm:px-4 py-2 flex justify-between items-center">
              <h2 className="text-base sm:text-lg font-bold">
                Transformer Visualization
              </h2>
            </div>

            {/* Top control - training mode toggle */}
            <div className="p-2 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-xs sm:text-sm font-medium text-gray-700">
                  Training:
                </span>
                <div className="flex items-center border border-gray-300 rounded-lg overflow-hidden shadow-sm w-28 sm:w-32">
                  <button
                    onClick={() => setTrainingMode(false)}
                    className={`flex-1 h-8 flex items-center justify-center text-xs sm:text-sm font-medium transition-colors border-r border-gray-300 ${
                      !trainingMode
                        ? 'bg-gray-200 text-gray-800 font-semibold'
                        : 'bg-white text-gray-500 hover:bg-gray-50'
                    }`}
                  >
                    <span className="block">Off</span>
                  </button>
                  <button
                    onClick={() => setTrainingMode(true)}
                    className={`flex-1 h-8 flex items-center justify-center text-xs sm:text-sm font-medium transition-colors ${
                      trainingMode
                        ? 'bg-blue-500 text-white font-semibold'
                        : 'bg-white text-gray-500 hover:bg-gray-50'
                    }`}
                  >
                    <span className="block">On</span>
                  </button>
                </div>
              </div>
              {/* Training status display */}
              {trainingMode && (
                <div className="flex items-center gap-4 text-xs sm:text-sm">
                  {targetTokenIndex !== null ? (
                    <>
                      <span className="text-gray-600">Target:</span>
                      <span className="font-mono font-bold text-green-600">
                        {vocabularyWords[targetTokenIndex]}
                      </span>
                      {trainingLoss !== null && (
                        <>
                          <span className="text-gray-600">Loss:</span>
                          <span className="font-mono font-bold text-blue-600">
                            {trainingLoss.toFixed(4)}
                          </span>
                        </>
                      )}
                    </>
                  ) : (
                    <span className="text-gray-500 italic">
                      Right-click a token to set target
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Tokenizer section */}
          <div className="mb-0.5 bg-white rounded p-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Tokenizer
            </h3>
            <div className="p-1 sm:p-2">
              <p className="text-[10px] sm:text-xs text-gray-600 mb-1 sm:mb-2">
                Click tokens to add to input sequence{trainingMode ? ' or right-click to set as target output' : ''} (hover to see
                embeddings)
              </p>
              <div className="flex flex-wrap gap-1 sm:gap-2">
                {vocabularyWords.map((word, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleTokenizerClick(idx)}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      if (trainingMode) {
                        setTargetTokenIndex(idx);
                      }
                    }}
                    className={`px-2 sm:px-3 py-1 sm:py-1.5 border ${
                      idx === recentlyAddedIndex
                        ? 'border-blue-500'
                        : idx === targetTokenIndex
                        ? 'border-green-400'
                        : 'border-gray-300'
                    } rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm ${
                      idx === targetTokenIndex ? 'bg-green-50' : 'bg-gray-100'
                    } hover:bg-gray-200 cursor-pointer font-mono transition-colors group relative`}
                  >
                    {word}
                    {/* Show embedding as matrix on hover */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                      <div className="mb-1 text-gray-600 text-center font-medium">
                        {word} embedding
                      </div>
                      <MatrixDisplay
                        data={[vocabularyEmbeddings[idx]]}
                        rowLabels={['']}
                        columnLabels={Array.from(
                          { length: embeddingDim },
                          (_, i) => `d${i + 1}`
                        )}
                        maxAbsValue={0.2}
                        cellSize="xs"
                        selectable={false}
                        matrixType="none"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Input Sequence section */}
          <div className="mb-0.5 bg-white rounded p-0.5">
            <div className="flex justify-between items-center border-b pb-0.5">
              <h3 className="text-xs sm:text-sm font-semibold">
                Input Sequence
              </h3>
              {/* Output token display */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] sm:text-xs text-gray-600">
                  Target Output:
                </span>
                {targetTokenIndex !== null ? (
                  <div
                    onClick={() => setTargetTokenIndex(null)}
                    className="px-2 sm:px-3 py-1 sm:py-1.5 border border-green-400 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-green-50 font-mono cursor-pointer hover:bg-red-50 hover:border-red-300 transition-colors"
                  >
                    {vocabularyWords[targetTokenIndex]}
                  </div>
                ) : (
                  <div className="px-2 sm:px-3 py-1 sm:py-1.5 border border-dashed border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center text-gray-400 italic">
                    none
                  </div>
                )}
              </div>
            </div>
            <div className="p-1 sm:p-2">
              <p className="text-[10px] sm:text-xs text-gray-600 mb-1 sm:mb-2">
                Click a token to remove it (hover to see embeddings)
              </p>
              {/* Input sequence area */}
              <div className="min-h-[40px] sm:min-h-[50px] border-2 border-dashed rounded-lg p-1 sm:p-2 transition-colors border-gray-300 bg-gray-50">
                {selectedTokenIndices.length === 0 ? (
                  <p className="text-gray-400 text-center text-xs sm:text-sm italic">
                    Drag tokens here...
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-1 sm:gap-2 relative">
                    {selectedTokenIndices.map((tokenIdx, seqIdx) => (
                      <div
                        key={seqIdx}
                        data-token-index={seqIdx}
                        onClick={() => handleSequenceTokenClick(seqIdx)}
                        className="px-2 sm:px-3 py-1 sm:py-1.5 border border-gray-400 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] h-7 sm:h-9 text-center shadow-sm bg-white font-mono cursor-pointer transition-all hover:bg-red-50 hover:border-red-300 group relative"
                      >
                        {vocabularyWords[tokenIdx]}
                        {/* Show embedding as matrix on hover */}
                        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                          <div className="mb-1 text-gray-600 text-center font-medium">
                            {vocabularyWords[tokenIdx]} embedding
                          </div>
                          <MatrixDisplay
                            data={[vocabularyEmbeddings[tokenIdx]]}
                            rowLabels={['']}
                            columnLabels={Array.from(
                              { length: embeddingDim },
                              (_, i) => `d${i + 1}`
                            )}
                            maxAbsValue={0.2}
                            cellSize="xs"
                            selectable={false}
                            matrixType="none"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="mb-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Embeddings with Positional Encoding
            </h3>
            <div
              className={`flex ${
                isMobile ? 'flex-col' : 'grid grid-cols-3'
              } lg:grid-cols-3 xl:grid-cols-12 gap-2 sm:gap-4 lg:gap-1`}
            >
              {/* Left: Raw Embeddings */}
              <div
                className={`${
                  isMobile ? 'w-full' : 'col-span-1'
                } xl:col-span-4 flex flex-col items-center`}
              >
                <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-1 sm:mb-0.5 text-center">
                  Raw Token Embeddings
                </h4>
                <div className="w-full overflow-x-auto pb-2">
                  <MatrixDisplay
                    data={rawEmbeddings}
                    rowLabels={tokenLabels}
                    columnLabels={Array.from(
                      { length: embeddingDim },
                      (_, i) => `d_${i + 1}`
                    )}
                    maxAbsValue={0.2}
                    cellSize="xs"
                    selectable={true}
                    selectedElement={
                      selectedElement?.matrixType === 'embeddings'
                        ? selectedElement
                        : null
                    }
                    matrixType="embeddings"
                    onElementClick={handleElementClick}
                    onValueChange={handleValueChange}
                    valueLabel={valueLabel}
                    autoOscillate={false}
                  />
                </div>
              </div>

              {/* Middle: Positional Encodings */}
              <div
                className={`${
                  isMobile ? 'w-full' : 'col-span-1'
                } xl:col-span-4 flex flex-col items-center`}
              >
                <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-1 sm:mb-0.5 text-center">
                  Positional Encodings
                </h4>
                <div className="w-full overflow-x-auto pb-2">
                  <MatrixDisplay
                    data={positionalEncodings.slice(0, tokenLabels.length)}
                    rowLabels={Array.from(
                      { length: tokenLabels.length },
                      (_, i) => `Pos ${i}`
                    )}
                    columnLabels={Array.from(
                      { length: embeddingDim },
                      (_, i) => `d_${i + 1}`
                    )}
                    maxAbsValue={0.2}
                    cellSize="xs"
                    selectable={false}
                    matrixType="none"
                  />
                </div>
              </div>

              {/* Right: Combined embeddings with positional encoding */}
              <div
                className={`${
                  isMobile ? 'w-full' : 'col-span-1'
                } xl:col-span-4 flex flex-col items-center`}
              >
                <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-1 sm:mb-0.5 text-center">
                  Embeddings + Pos. Encoding
                  {trainingMode ? ` + Dropout(${embeddingDropoutRate})` : ''}
                </h4>
                <div className="w-full overflow-x-auto pb-2">
                  <MatrixDisplay
                    data={embeddingsWithDropout}
                    rowLabels={tokenLabels}
                    columnLabels={Array.from(
                      { length: embeddingDim },
                      (_, i) => `d_${i + 1}`
                    )}
                    maxAbsValue={0.2}
                    cellSize="xs"
                    selectable={false}
                    matrixType="none"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mb-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Self-Attention
            </h3>

            <AttentionHead
              embeddings={embeddingsWithDropout} // Using embeddings with positional encoding and dropout
              weightQ={attentionWeights.weightQ}
              weightK={attentionWeights.weightK}
              weightV={attentionWeights.weightV}
              tokenLabels={tokenLabels}
              showSteps={true}
              onContextComputed={handleAttentionContextComputed}
              selectedElement={selectedElement}
              onElementClick={handleElementClick}
              onValueChange={handleValueChange}
              dropoutRate={attentionDropoutRate}
              applyTrainingDropout={trainingMode}
              dropoutCycle={trainingCycle} // Pass the training cycle to force updates
            />
          </div>

          <div className="mt-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
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
                selectedElement={selectedElement}
                onElementClick={handleElementClick}
                onValueChange={handleValueChange}
                dropoutRate={ffnDropoutRate}
                applyTrainingDropout={trainingMode}
                activationFn={relu} // ReLU activation as default
                activationFnName="ReLU"
                dropoutCycle={trainingCycle} // Pass the training cycle to force updates
                onOutputComputed={handleFfnOutputComputed}
              />
            ) : (
              <div className="p-0.5 bg-gray-100 rounded">
                <p className="text-gray-600 italic text-[0.5rem] sm:text-[0.6rem] text-center py-1">
                  Waiting for attention computation...
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Next Token Prediction Section */}
        {ffnOutput.length > 0 && (
          <div className="mt-0.5 mb-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Next Token Prediction
            </h3>
            <div className="bg-white rounded p-0.5">
              <div
                className={`flex ${
                  isMobile ? 'flex-col' : 'grid grid-cols-3'
                } lg:grid-cols-3 xl:grid-cols-12 gap-2 sm:gap-3 lg:gap-1`}
              >
                {/* Calculate token similarities and predictions */}
                {(() => {
                  // Get the next token prediction vector (last token's embedding)
                  const nextTokenPrediction = ffnOutput[ffnOutput.length - 1];

                  // Calculate dot product similarity with each vocabulary token
                  const dotProducts = vocabularyEmbeddings.map(
                    (vocabEmbedding) =>
                      vectorDotProduct(nextTokenPrediction, vocabEmbedding)
                  );

                  // Apply softmax to the dot products to get probability-like values
                  // First find the maximum for numerical stability
                  const maxDotProduct = Math.max(...dotProducts);
                  // Calculate exp(x - max) for each dot product
                  const expValues = dotProducts.map((dp) =>
                    Math.exp(dp - maxDotProduct)
                  );
                  // Sum of all exp values
                  const sumExp = expValues.reduce((a, b) => a + b, 0);
                  // Normalize to get softmax values
                  const softmaxValues = expValues.map((exp) => exp / sumExp);

                  // Create pairs of [index, softmax value] so we can sort them while keeping the original indices
                  const indexedSoftmax = softmaxValues.map((value, index) => ({
                    index,
                    value,
                  }));
                  // Sort by softmax value in descending order
                  const sortedSoftmax = [...indexedSoftmax].sort(
                    (a, b) => b.value - a.value
                  );

                  // Get corresponding token labels in the sorted order
                  const sortedTokenLabels = sortedSoftmax.map(
                    (item) => vocabularyWords[item.index]
                  );

                  // Get the highest probability token (first one in sorted list)
                  const topPredictedTokenIndex = sortedSoftmax[0].index;
                  const topPredictedToken =
                    vocabularyWords[topPredictedTokenIndex];
                  const topPredictedTokenEmbedding =
                    vocabularyEmbeddings[topPredictedTokenIndex];

                  return (
                    <>
                      {/* Next Token Prediction Vector */}
                      <div
                        className={`${
                          isMobile ? 'w-full' : 'col-span-1'
                        } xl:col-span-3 flex flex-col items-center`}
                      >
                        <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-0.5 text-center">
                          Next Token Vector
                        </h4>
                        {/* Use the last token's embedding as the prediction for the next token */}
                        <MatrixDisplay
                          data={[nextTokenPrediction]} // Use the last token's embedding
                          rowLabels={['']}
                          columnLabels={Array.from(
                            { length: embeddingDim },
                            (_, i) => `d_${i + 1}`
                          )}
                          maxAbsValue={0.3}
                          cellSize="xs"
                          selectable={false}
                          matrixType="none"
                        />
                      </div>

                      {/* Similarity Scores */}
                      <div
                        className={`${
                          isMobile ? 'w-full' : 'col-span-1'
                        } xl:col-span-6 flex flex-col items-center`}
                      >
                        <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-0.5 text-center">
                          Token Similarities
                        </h4>
                        <div className="w-full">
                          {/* Dot Products */}
                          <div className="mb-2">
                            <h5 className="text-[0.55rem] sm:text-[0.6rem] font-medium mb-0.5 text-center">
                              Dot Product
                            </h5>
                            <div className="overflow-x-auto">
                              <MatrixDisplay
                                data={[dotProducts]}
                                rowLabels={['']}
                                columnLabels={vocabularyWords}
                                maxAbsValue={
                                  Math.max(
                                    ...dotProducts.map((dp) => Math.abs(dp))
                                  ) || 0.3
                                }
                                cellSize="xs"
                                selectable={false}
                                matrixType="none"
                              />
                            </div>
                          </div>

                          {/* Softmax Values (sorted by value, largest first) */}
                          <div>
                            <h5 className="text-[0.55rem] sm:text-[0.6rem] font-medium mb-0.5 text-center">
                              Softmax
                            </h5>
                            <div className="overflow-x-auto">
                              <MatrixDisplay
                                data={[sortedSoftmax.map((item) => item.value)]}
                                rowLabels={['']}
                                columnLabels={sortedTokenLabels}
                                maxAbsValue={1.0} // Softmax values are between 0 and 1
                                cellSize="xs"
                                selectable={false}
                                matrixType="none"
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Most Likely Token */}
                      <div
                        className={`${
                          isMobile ? 'w-full' : 'col-span-1'
                        } xl:col-span-3 flex flex-col items-center`}
                      >
                        <h4 className="text-[0.6rem] sm:text-[0.65rem] font-medium mb-1 sm:mb-2 text-center">
                          Most Likely Next Token
                        </h4>
                        <div className="w-full flex flex-col items-center">
                          <div className="flex justify-center">
                            {/* Token styled like tokenizer tokens */}
                            <div className="px-2 sm:px-3 py-1 sm:py-1.5 border border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-gray-100 font-mono cursor-pointer group relative hover:bg-gray-200 transition-colors">
                              {topPredictedToken}
                              {/* Show embedding as matrix on hover */}
                              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                                <div className="mb-1 text-gray-600 text-center font-medium">
                                  {topPredictedToken} embedding
                                </div>
                                <MatrixDisplay
                                  data={[topPredictedTokenEmbedding]}
                                  rowLabels={['']}
                                  columnLabels={Array.from(
                                    { length: embeddingDim },
                                    (_, i) => `d${i + 1}`
                                  )}
                                  maxAbsValue={0.2}
                                  cellSize="xs"
                                  selectable={false}
                                  matrixType="none"
                                />
                              </div>
                            </div>
                          </div>
                          <div className="text-[0.6rem] sm:text-[0.65rem] font-mono text-blue-600 bg-blue-50 px-2 sm:px-3 py-0.5 rounded-md border border-blue-100 min-w-[55px] sm:min-w-[60px] text-center">
                            p={sortedSoftmax[0].value.toFixed(4)}
                          </div>

                          {/* Raw embedding for the predicted token */}
                          <h5 className="text-[0.6rem] sm:text-[0.65rem] font-medium mt-1 sm:mt-2 mb-1 text-center text-gray-700 pt-1 sm:pt-1.5 w-full">
                            Raw Token Embedding
                          </h5>
                          <MatrixDisplay
                            data={[
                              vocabularyEmbeddings[topPredictedTokenIndex],
                            ]}
                            rowLabels={[topPredictedToken]}
                            columnLabels={Array.from(
                              { length: embeddingDim },
                              (_, i) => `d_${i + 1}`
                            )}
                            maxAbsValue={0.2}
                            cellSize="xs"
                            selectable={false}
                            matrixType="none"
                          />
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        )}

        <div className="bg-white rounded p-0.5 mt-4 sm:mt-8 text-[0.55rem] sm:text-[0.6rem]">
          <div className="flex flex-col gap-1">
            <p className="text-gray-700">
              Blue: positive, Red: negative. Click a value to edit (magenta
              border).
              {trainingMode &&
                (targetTokenIndex !== null 
                  ? ` Training mode: learning to predict "${vocabularyWords[targetTokenIndex]}".`
                  : ' Training mode enabled: select a target token (right-click in tokenizer).')}
              {trainingMode && targetTokenIndex !== null && trainingLoss !== null && (
                <span className="ml-2 text-blue-600">
                  Loss: {trainingLoss.toFixed(4)}
                </span>
              )}
            </p>

            <div className="flex flex-wrap gap-2 sm:gap-4">
              <div className="flex items-center">
                <span className="font-semibold mr-1">Tokens:</span>
                <span>{tokenLabels.length}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">Embedding Dim:</span>
                <span>{embeddingDim}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">Attention Head Dim:</span>
                <span>{attentionHeadDim}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">FFN Hidden Dim:</span>
                <span>{mlpHiddenDim}</span>
              </div>

              {trainingMode && (
                <>
                  <div className="flex items-center">
                    <span className="font-semibold mr-1">
                      Embedding Dropout:
                    </span>
                    <span>{(embeddingDropoutRate * 100).toFixed(0)}%</span>
                  </div>

                  <div className="flex items-center">
                    <span className="font-semibold mr-1">
                      Attention Dropout:
                    </span>
                    <span>{(attentionDropoutRate * 100).toFixed(0)}%</span>
                  </div>

                  <div className="flex items-center">
                    <span className="font-semibold mr-1">FFN Dropout:</span>
                    <span>{(ffnDropoutRate * 100).toFixed(0)}%</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
