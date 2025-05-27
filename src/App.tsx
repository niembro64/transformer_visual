import { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import HistoryGraph from './components/HistoryGraph';
import MatrixDisplay from './components/MatrixDisplay';
import SoftmaxHistoryGraph from './components/SoftmaxHistoryGraph';
import {
  addPositionalEncodings,
  generatePositionalEncodings,
  generateSampleAttentionWeights,
  generateSampleEmbeddings,
  generateSampleMLPWeights,
  isPortraitOrientation,
  relu,
  vectorDotProduct,
} from './utils/matrixOperations';

export type HistoryTrainingEntry = {
  loss: number;
  targetToken: string;
  predictedToken: string;
  targetProb: string;
};

export type HistorySoftMaxEntry = {
  softmaxValues: { token: string; probability: number }[];
  timestamp: number;
};

// Training configuration constants
const TRAINING_INTERVAL_MS = 0.01;
const EXPONENTIAL_DECIMALS = 4; // Number of decimal places for exponential values
const DIM_EMBEDDING = isPortraitOrientation() ? 6 : 10; // Dimension of embeddings (d_model)
const DIM_ATTENTION_HEAD = isPortraitOrientation() ? 2 : 4; // Dimension of attention heads (d_k = d_v = d_model / num_heads)
const DIM_MLP_HIDDEN = isPortraitOrientation() ? 6 : 8; // Dimension of MLP hidden layer (d_ff = 8, typically 4x d_model)

function App() {
  // Fixed dimension values
  const HISTORY_DISPLAY_STEPS = isPortraitOrientation() ? 50 : 100; // Number of training steps to show in history graph

  // Training mode - determines if dropout is applied and weights are updated
  const [trainingMode, setTrainingMode] = useState(true);
  // Target output token for training (what we're trying to predict)
  const [targetTokenIndex, setTargetTokenIndex] = useState<number | null>(null);
  // Learning rate for gradient descent
  const [learningRate, setLearningRate] = useState(
    isPortraitOrientation() ? 0.001 : 0.001
  );

  const [historyTraining, setHistoryTraining] = useState<
    HistoryTrainingEntry[]
  >([]); // History of training logs

  const [historySoftMax, setHistorySoftMax] = useState<HistorySoftMaxEntry[]>(
    []
  ); // History of softmax probabilities

  // Total number of training steps (not limited by history display)
  const [totalTrainingSteps, setTotalTrainingSteps] = useState(0);

  // Track if device is in mobile mode (height > width)
  const [isMobile, setIsMobile] = useState(isPortraitOrientation);

  // Vocabulary of 25 common words
  // eslint-disable-next-line react-hooks/exhaustive-deps

  let vocabularyWords: string[] = [];
  if (isPortraitOrientation()) {
    vocabularyWords = ['lore', 'ipsu', 'dolo', 'sit', 'amet', 'cons'];
  } else {
    vocabularyWords = [
      'lore',
      'ipsu',
      'dolo',
      'sit',
      'amet',
      'cons',
      'adip',
      'elit',
      'sed',
      'do',
      'eius',
      'temp',
      'inci',
      'ut',
      'labo',
    ];
  }

  // Generate vocabulary embeddings - mutable state
  const [vocabularyEmbeddings, setVocabularyEmbeddings] = useState(() =>
    generateSampleEmbeddings(vocabularyWords.length, DIM_EMBEDDING)
  );

  const initInputSequence: number[] = isPortraitOrientation()
    ? [2, 3, 1]
    : [3, 4, 2, 1, 0, 6, 7];

  // Track selected tokens (indices into vocabulary)
  const [selectedTokenIndices, setSelectedTokenIndices] =
    useState<number[]>(initInputSequence);

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
    generatePositionalEncodings(maxSeqLength, DIM_EMBEDDING)
  );

  // Update positional encodings when dimensions change
  useEffect(() => {
    setPositionalEncodings(
      generatePositionalEncodings(maxSeqLength, DIM_EMBEDDING)
    );
  }, [maxSeqLength, DIM_EMBEDDING]);

  // Get embeddings for selected tokens
  const rawEmbeddings = useMemo(() => {
    return selectedTokenIndices.map((idx) => [...vocabularyEmbeddings[idx]]);
  }, [selectedTokenIndices, vocabularyEmbeddings]);

  // Update vocabulary embeddings when dimension changes
  useEffect(() => {
    setVocabularyEmbeddings(
      generateSampleEmbeddings(vocabularyWords.length, DIM_EMBEDDING)
    );
  }, [vocabularyWords.length]);

  // Apply positional encodings to embeddings
  const embeddings = useMemo(
    () => addPositionalEncodings(rawEmbeddings, positionalEncodings),
    [rawEmbeddings, positionalEncodings]
  );

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
          const expValues = dotProducts.map((dp) =>
            Math.exp(dp - maxDotProduct)
          );
          const sumExp = expValues.reduce((a, b) => a + b, 0);
          const probabilities = expValues.map((exp) => exp / sumExp);

          // Cross-entropy loss: -log(probability of correct token)
          const targetProb = probabilities[targetTokenIndex];
          const loss = -Math.log(Math.max(targetProb, 1e-7)); // Avoid log(0)
          setTrainingLoss(loss);

          // Convert gradient to embedding space
          // We need to push the output embedding towards the target embedding
          const targetEmbedding = vocabularyEmbeddings[targetTokenIndex];
          const outputGradient = predictedEmbedding.map(
            (val, i) => val - targetEmbedding[i]
          );

          const predictedTokenIndex = dotProducts.indexOf(
            Math.max(...dotProducts)
          );

          const historyItem: HistoryTrainingEntry = {
            loss: loss,
            targetToken: vocabularyWords[targetTokenIndex],
            predictedToken: vocabularyWords[predictedTokenIndex],
            targetProb: targetProb.toFixed(EXPONENTIAL_DECIMALS),
          };

          // Increment total training steps
          setTotalTrainingSteps((prev) => prev + 1);

          // Add to training history (limited to HISTORY_DISPLAY_STEPS)
          setHistoryTraining((prev: HistoryTrainingEntry[]) => {
            const newHistory = [...prev, historyItem];
            // Keep only the last HISTORY_DISPLAY_STEPS entries
            return newHistory.slice(-HISTORY_DISPLAY_STEPS);
          });

          // Save softmax probabilities to history
          const softmaxEntry: HistorySoftMaxEntry = {
            softmaxValues: vocabularyWords.map((word, idx) => ({
              token: word,
              probability: probabilities[idx],
            })),
            timestamp: Date.now(),
          };

          setHistorySoftMax((prev: HistorySoftMaxEntry[]) => {
            const newHistory = [...prev, softmaxEntry];
            // Keep only the last HISTORY_DISPLAY_STEPS entries
            return newHistory.slice(-HISTORY_DISPLAY_STEPS);
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
                    const avgError =
                      outputGradient.reduce((a, b) => a + b, 0) /
                      outputGradient.length;
                    newWeights.W1[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.W1.length; i++) {
                for (let j = 0; j < newWeights.W1[i].length; j++) {
                  // Simplified gradient
                  const avgError =
                    outputGradient.reduce((a, b) => a + b, 0) /
                    outputGradient.length;
                  newWeights.W1[i][j] -= learningRate * avgError * 0.5; // Increased from 0.01
                }
              }
            }

            // Update b1 biases
            const avgError =
              outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;
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
            const avgError =
              outputGradient.reduce((a, b) => a + b, 0) / outputGradient.length;

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

          // Note: In a real transformer, token embeddings would typically remain fixed
          // during training, with only the model weights being updated
        }
      }, TRAINING_INTERVAL_MS);
    }

    return () => {
      if (timerId !== null) {
        window.clearInterval(timerId);
      }
    };
  }, [
    trainingMode,
    weightUpdateStepSize,
    targetTokenIndex,
    ffnOutput,
    selectedElement,
    learningRate,
    vocabularyEmbeddings,
    vocabularyWords,
  ]);

  // Use embeddings directly (no dropout)
  const embeddingsWithDropout = embeddings;

  // Sample data generation for attention weights - regenerate when dimensions change
  const [attentionWeights, setAttentionWeights] = useState(() =>
    generateSampleAttentionWeights(DIM_EMBEDDING, DIM_ATTENTION_HEAD)
  );

  // Update attention weights when dimensions change
  useEffect(() => {
    setAttentionWeights(
      generateSampleAttentionWeights(DIM_EMBEDDING, DIM_ATTENTION_HEAD)
    );
  }, [DIM_EMBEDDING, DIM_ATTENTION_HEAD]);

  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights, setMlpWeights] = useState(() =>
    generateSampleMLPWeights(DIM_EMBEDDING, DIM_MLP_HIDDEN, DIM_ATTENTION_HEAD)
  );

  // Update MLP weights when dimensions change
  useEffect(() => {
    setMlpWeights(
      generateSampleMLPWeights(
        DIM_EMBEDDING,
        DIM_MLP_HIDDEN,
        DIM_ATTENTION_HEAD
      )
    );
  }, [DIM_EMBEDDING, DIM_MLP_HIDDEN, DIM_ATTENTION_HEAD]);

  // No need for these functions anymore - handled by drag and drop

  // Click interaction state
  const [recentlyAddedIndex, setRecentlyAddedIndex] = useState<number | null>(
    null
  );

  // Handler for tokenizer click - adds to sequence or sets target based on training mode
  const handleTokenizerClick = useCallback(
    (index: number) => {
      if (trainingMode) {
        // In training mode, set as target output
        setTargetTokenIndex(index);
      } else {
        // In inference mode, add to input sequence
        setSelectedTokenIndices((prev) => [...prev, index]);
        setRecentlyAddedIndex(index);

        // Clear the highlight after a brief delay
        setTimeout(() => {
          setRecentlyAddedIndex(null);
        }, 500);
      }
    },
    [trainingMode]
  );

  // Handler to remove a token from the input sequence
  const handleSequenceTokenClick = useCallback((index: number) => {
    setSelectedTokenIndices((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // No need for dimension adjustment functions anymore

  const initialElement: ElementObject | null = useMemo(() => {
    // Initialize selectedElement to the second-to-last token's 4th embedding
    if (selectedTokenIndices.length > 0 && DIM_EMBEDDING > 0) {
      const tokenIndex = Math.max(0, selectedTokenIndices.length - 2); // Second-to-last token
      const embIndex = Math.min(3, DIM_EMBEDDING - 1); // 4th embedding (index 3) or last if fewer

      return {
        matrixType: 'embeddings',
        row: tokenIndex,
        col: embIndex,
      };
    }
    return null;
  }, [selectedTokenIndices, DIM_EMBEDDING]);

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
          if (row < selectedTokenIndices.length && col < DIM_EMBEDDING) {
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
    [
      selectedElement,
      selectedTokenIndices,
      vocabularyEmbeddings,
      attentionWeights.weightQ,
      attentionWeights.weightK,
      attentionWeights.weightV,
      mlpWeights.W1,
      mlpWeights.b1,
      mlpWeights.W2,
      mlpWeights.b2,
    ]
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
          <div
            className={`mb-4 ${
              trainingMode
                ? 'bg-gradient-to-r from-green-50 to-emerald-50'
                : 'bg-gradient-to-r from-blue-50 to-indigo-50'
            } rounded-lg shadow-sm overflow-hidden`}
          >
            {/* Header bar */}
            <div
              className={`${
                trainingMode
                  ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                  : 'bg-gradient-to-r from-blue-500 to-indigo-600'
              } text-white px-2 sm:px-4 py-2 flex justify-between items-center`}
            >
              <h2 className="text-base sm:text-lg font-bold">
                Transformer Visualization
              </h2>
            </div>

            {/* Top control - mode toggle */}
            <div className="p-2 flex gap-2 justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-xs sm:text-sm font-medium text-gray-700">
                  Mode:
                </span>
                <button
                  onClick={() => setTrainingMode(!trainingMode)}
                  className={`px-4 sm:px-6 h-8 flex items-center justify-center text-xs sm:text-sm font-medium transition-all duration-200 rounded-lg shadow-sm border ${
                    trainingMode
                      ? 'bg-green-500 hover:bg-green-600 text-white border-green-600'
                      : 'bg-blue-500 hover:bg-blue-600 text-white border-blue-600'
                  }`}
                >
                  <span className="block">
                    {trainingMode ? 'Training' : 'Inferencing'}
                  </span>
                </button>
              </div>
              {/* Training controls */}
              {trainingMode && (
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <label
                      htmlFor="learningRate"
                      className="text-xs sm:text-sm font-medium text-gray-700"
                    >
                      Learning Rate:
                    </label>
                    <input
                      type="number"
                      value={learningRate}
                      onChange={(e) =>
                        setLearningRate(parseFloat(e.target.value) || 0)
                      }
                      step="0.001"
                      min="0"
                      max="1"
                      className="w-20 sm:w-24 px-2 py-1 text-xs sm:text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                  </div>
                  {targetTokenIndex === null && (
                    <span className="text-xs sm:text-sm text-gray-500 italic">
                      Click a token to set target
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
                Click tokens to{' '}
                {trainingMode
                  ? 'set as target output'
                  : 'add to input sequence'}{' '}
                (hover to see embeddings)
              </p>
              <div className="flex flex-wrap gap-1 sm:gap-2">
                {vocabularyWords.map((word, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleTokenizerClick(idx)}
                    className={`px-2 sm:px-3 py-1 sm:py-1.5 ${
                      trainingMode && idx === targetTokenIndex
                        ? 'border-2 border-green-500'
                        : `border ${
                            idx === recentlyAddedIndex
                              ? 'border-blue-500'
                              : 'border-gray-300'
                          }`
                    } rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm ${
                      trainingMode && idx === targetTokenIndex
                        ? 'bg-green-100 text-green-900 font-semibold hover:bg-green-200'
                        : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                    } cursor-pointer font-mono transition-colors group relative`}
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
                          { length: DIM_EMBEDDING },
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
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Input Sequence
            </h3>
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
                        className="px-2 sm:px-3 py-1 sm:py-1.5 border border-purple-400 bg-purple-100 text-purple-900 hover:bg-purple-200 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] h-7 sm:h-9 text-center shadow-sm font-mono cursor-pointer transition-all group relative"
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
                              { length: DIM_EMBEDDING },
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

          {/* Output Token section */}
          <div className="mb-0.5 bg-white rounded p-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Output Token
            </h3>
            <div className="p-1 sm:p-2">
              <div className="flex flex-col sm:flex-row gap-2 sm:gap-4">
                {/* Next Token Prediction */}
                <div className="flex-1">
                  <p className="text-[10px] sm:text-xs text-gray-600 mb-1">
                    Currently Predicted Next Token:
                  </p>
                  {ffnOutput.length > 0 ? (
                    (() => {
                      const nextTokenPrediction =
                        ffnOutput[ffnOutput.length - 1];
                      const dotProducts = vocabularyEmbeddings.map(
                        (vocabEmbedding) =>
                          vectorDotProduct(nextTokenPrediction, vocabEmbedding)
                      );
                      const predictedTokenIndex = dotProducts.indexOf(
                        Math.max(...dotProducts)
                      );
                      const maxDotProduct = Math.max(...dotProducts);
                      const expValues = dotProducts.map((dp) =>
                        Math.exp(dp - maxDotProduct)
                      );
                      const sumExp = expValues.reduce((a, b) => a + b, 0);
                      const probabilities = expValues.map(
                        (exp) => exp / sumExp
                      );
                      const predictedProb = probabilities[predictedTokenIndex];

                      return (
                        <div className="flex items-center gap-2">
                          <div className="px-2 sm:px-3 py-1 sm:py-1.5 border-2 border-blue-500 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-blue-100 font-mono group relative cursor-pointer text-blue-900 font-semibold">
                            {vocabularyWords[predictedTokenIndex]}
                            {/* Show embedding as matrix on hover */}
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                              <div className="mb-1 text-gray-600 text-center font-medium">
                                {vocabularyWords[predictedTokenIndex]} embedding
                              </div>
                              <MatrixDisplay
                                data={[
                                  vocabularyEmbeddings[predictedTokenIndex],
                                ]}
                                rowLabels={['']}
                                columnLabels={Array.from(
                                  { length: DIM_EMBEDDING },
                                  (_, i) => `d${i + 1}`
                                )}
                                maxAbsValue={0.2}
                                cellSize="xs"
                                selectable={false}
                                matrixType="none"
                              />
                            </div>
                          </div>
                          <span className="text-[10px] sm:text-xs text-gray-600 font-mono">
                            p: {predictedProb >= 0 ? '+' : ''}
                            {predictedProb.toExponential(EXPONENTIAL_DECIMALS)}
                          </span>
                        </div>
                      );
                    })()
                  ) : (
                    <div className="px-2 sm:px-3 py-1 sm:py-1.5 border border-dashed border-gray-300 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center text-gray-400 italic">
                      computing...
                    </div>
                  )}
                </div>

                {/* Sorted Softmax Token Similarities */}
                {ffnOutput.length > 0 && (
                  <div className="mt-3 mb-3">
                    <p className="text-[10px] sm:text-xs text-gray-600 mb-1">
                      Token Probabilities (sorted):
                    </p>
                    {(() => {
                      const nextTokenPrediction =
                        ffnOutput[ffnOutput.length - 1];
                      const dotProducts = vocabularyEmbeddings.map(
                        (vocabEmbedding) =>
                          vectorDotProduct(nextTokenPrediction, vocabEmbedding)
                      );
                      const maxDotProduct = Math.max(...dotProducts);
                      const expValues = dotProducts.map((dp) =>
                        Math.exp(dp - maxDotProduct)
                      );
                      const sumExp = expValues.reduce((a, b) => a + b, 0);
                      const probabilities = expValues.map(
                        (exp) => exp / sumExp
                      );

                      // Create indexed pairs and sort by value
                      const indexed = probabilities.map((value, index) => ({
                        index,
                        value,
                      }));
                      const sortedSoftmax = indexed.sort(
                        (a, b) => b.value - a.value
                      );
                      const sortedTokenLabels = sortedSoftmax.map(
                        (item) => vocabularyWords[item.index]
                      );

                      return (
                        <div className="overflow-x-auto">
                          <MatrixDisplay
                            data={[sortedSoftmax.map((item) => item.value)]}
                            rowLabels={['']}
                            columnLabels={sortedTokenLabels}
                            maxAbsValue={1.0}
                            cellSize="xs"
                            selectable={false}
                            matrixType="none"
                          />
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* Target Output (Training Mode Only) */}
                {trainingMode && (
                  <div className="flex-1">
                    <p className="text-[10px] sm:text-xs text-gray-600 mb-1">
                      Desired Next Token:
                    </p>
                    <div className="min-h-[40px] sm:min-h-[50px] border-2 border-dashed rounded-lg p-1 sm:p-2 transition-colors border-gray-300 bg-gray-50">
                      {targetTokenIndex !== null ? (
                        <div className="flex items-center gap-2">
                          <div className="px-2 sm:px-3 py-1 sm:py-1.5 border-2 border-green-500 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-green-100 font-mono group relative cursor-pointer text-green-900 font-semibold">
                            {vocabularyWords[targetTokenIndex]}
                            {/* Show embedding as matrix on hover */}
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-white border border-gray-200 text-gray-700 text-[10px] rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50">
                              <div className="mb-1 text-gray-600 text-center font-medium">
                                {vocabularyWords[targetTokenIndex]} embedding
                              </div>
                              <MatrixDisplay
                                data={[vocabularyEmbeddings[targetTokenIndex]]}
                                rowLabels={['']}
                                columnLabels={Array.from(
                                  { length: DIM_EMBEDDING },
                                  (_, i) => `d${i + 1}`
                                )}
                                maxAbsValue={0.2}
                                cellSize="xs"
                                selectable={false}
                                matrixType="none"
                              />
                            </div>
                          </div>
                          {trainingLoss !== null && (
                            <span className="text-[10px] sm:text-xs text-gray-600 font-mono">
                              Loss: {trainingLoss >= 0 ? '+' : ''}
                              {trainingLoss.toExponential(EXPONENTIAL_DECIMALS)}
                            </span>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-center text-xs sm:text-sm italic">
                          Click a token in tokenizer to set target
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Training Status Message */}
              {trainingMode && targetTokenIndex === null && (
                <p className="text-[10px] sm:text-xs text-gray-500 italic mt-2">
                  Click a token in the tokenizer to set as target output
                </p>
              )}
            </div>
          </div>

          {/* History Graphs */}
          {trainingMode && (
            <>
              <SoftmaxHistoryGraph
                history={historySoftMax}
                maxPoints={HISTORY_DISPLAY_STEPS}
                vocabularyWords={vocabularyWords}
                totalSteps={totalTrainingSteps}
              />
              <HistoryGraph
                history={historyTraining}
                maxPoints={HISTORY_DISPLAY_STEPS}
                totalSteps={totalTrainingSteps}
              />
            </>
          )}

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
                      { length: DIM_EMBEDDING },
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
                    isTrainingMode={trainingMode}
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
                      { length: DIM_EMBEDDING },
                      (_, i) => `d_${i + 1}`
                    )}
                    maxAbsValue={1}
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
                </h4>
                <div className="w-full overflow-x-auto pb-2">
                  <MatrixDisplay
                    data={embeddingsWithDropout}
                    rowLabels={tokenLabels}
                    columnLabels={Array.from(
                      { length: DIM_EMBEDDING },
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
                activationFn={relu} // ReLU activation as default
                activationFnName="ReLU"
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
                            { length: DIM_EMBEDDING },
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
                            <div className="px-2 sm:px-3 py-1 sm:py-1.5 border-2 border-blue-500 rounded text-xs sm:text-sm min-w-[2.5rem] sm:min-w-[3.5rem] text-center shadow-sm bg-blue-100 font-mono cursor-pointer group relative text-blue-900 font-semibold">
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
                                    { length: DIM_EMBEDDING },
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
                          <div className="text-[0.6rem] sm:text-[0.65rem] font-mono text-gray-600 px-2 sm:px-3 py-0.5 min-w-[55px] sm:min-w-[60px] text-center">
                            p:{' '}
                            {sortedSoftmax[0]?.value
                              ? (sortedSoftmax[0].value >= 0 ? '+' : '') +
                                sortedSoftmax[0].value.toExponential(2)
                              : '+0.00e+0'}
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
                              { length: DIM_EMBEDDING },
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
            </p>

            <div className="flex flex-wrap gap-2 sm:gap-4">
              <div className="flex items-center">
                <span className="font-semibold mr-1">Tokens:</span>
                <span>{tokenLabels.length}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">Embedding Dim:</span>
                <span>{DIM_EMBEDDING}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">Attention Head Dim:</span>
                <span>{DIM_ATTENTION_HEAD}</span>
              </div>

              <div className="flex items-center">
                <span className="font-semibold mr-1">FFN Hidden Dim:</span>
                <span>{DIM_MLP_HIDDEN}</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
