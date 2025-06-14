import { useCallback, useEffect, useMemo, useState } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import HistoryGraphLoss from './components/HistoryGraphLoss';
import MatrixDisplay from './components/MatrixDisplay';
import HistoryGraphSoftmax from './components/HistoryGraphSoftmax';
import Token from './components/Token';
import {
  addPositionalEncodings,
  generatePositionalEncodings,
  generateSampleAttentionWeights,
  generateSampleEmbeddings,
  generateSampleMLPWeights,
  isPortraitOrientation,
  relu,
  vectorDotProduct,
  hasInvalidValues,
  hasInvalidValuesVector,
  getMatrixErrorDetails,
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

const TRAINING_INTERVAL_MS = process.env.NODE_ENV === 'development' ? 30 : 30;
const EXPONENTIAL_DECIMALS = 4;

const dimValDesktop = 8;

const dimValMobile = 4;
const dimValMobileSmall = 3

const DIM_EMBEDDING = isPortraitOrientation() ? dimValMobile : dimValDesktop; // 6
const DIM_ATTENTION_HEAD = isPortraitOrientation()
  ? dimValMobileSmall
  : dimValDesktop; // 6
const DIM_MLP_HIDDEN = isPortraitOrientation() ? dimValMobileSmall : dimValDesktop; // 6
const ATTENTION_LR_MULTIPLIER = 1.0;
const EMBEDDING_STRENGTH_MULTIPLIER = 5;

function App() {
  // Fixed dimension values
  const HISTORY_DISPLAY_STEPS = isPortraitOrientation() ? 200 : 300; // Number of training steps to show in history graph

  // Error state to track calculation errors
  const [calculationError, setCalculationError] = useState<string | null>(null);

  // Training mode - determines if weights are updated and which training method is used
  type TrainingMode = 'Inferencing' | 'Train-One' | 'Train-All';
  const [trainingMode, setTrainingMode] = useState<TrainingMode>('Train-One');
  // Target output token for training (what we're trying to predict)
  const [targetTokenIndex, setTargetTokenIndex] = useState<number | null>(null);
  // Learning rate for gradient descent
  const [learningRate, setLearningRate] = useState(
    isPortraitOrientation() ? 0.002 : 0.005
  );

  // Whether Train-All mode is actively running
  const [isTrainAllRunning, setIsTrainAllRunning] = useState(false);

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

  const vocabularyWords: string[] = useMemo(() => {
    if (isMobile) {
      return [
        // 'hi', // 0
        // 'car', // 'bot', // 1
        'yo', // 'run', // 3
        'brr', // 5
        'ai', // 2
        'go', // 6
        'bot', // 'car', // 4
        'zzz', // 7
        // 'dog', // 8
        // 'run', //'go', // 9
        // 'zzz', //10
        // 'id', //11
        // 'do', //12
        // 'cat', //13
        // 'up', //14
        // 'lol', //15
        // 'the', //16
      ];
    }
    return [
      'hi', // 0
      'car', // 'bot', // 1
      'ai', // 2
      'go', // 'run', // 3
      'bot', // 'car', // 4
      'brr', // 5
      'yo', // 6
      'big', // 7
      'dog', // 8
      'run', //'go', // 9
      'zzz', //10
      'id', //11
      'do', //12
      'cat', //13
      'up', //14
      'lol', //15
      'the', //16
    ];
  }, []);

  const [vocabularyEmbeddings, setVocabularyEmbeddings] = useState(() =>
    generateSampleEmbeddings(
      vocabularyWords.length,
      DIM_EMBEDDING,
      EMBEDDING_STRENGTH_MULTIPLIER
    )
  );

  // Input sequence to say: "big AI bot go brr"
  const initInputSequence: number[] = [2, 4, 3];
  // 'big' + 'ai' + 'bot' + 'go' + 'brr'
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
      generateSampleEmbeddings(
        vocabularyWords.length,
        DIM_EMBEDDING,
        EMBEDDING_STRENGTH_MULTIPLIER
      )
    );
  }, [vocabularyWords.length]);

  // Apply positional encodings to embeddings
  const embeddings = useMemo(() => {
    try {
      const result = addPositionalEncodings(rawEmbeddings, positionalEncodings);

      // Check for invalid values
      if (hasInvalidValues(result)) {
        const error = getMatrixErrorDetails(result, 'embeddings');
        setCalculationError(error || 'Invalid values detected in embeddings');
        return rawEmbeddings; // Return raw embeddings as fallback
      }

      return result;
    } catch (error) {
      setCalculationError(`Error in positional encoding: ${error}`);
      return rawEmbeddings;
    }
  }, [rawEmbeddings, positionalEncodings]);

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

    const shouldRunTraining =
      (trainingMode === 'Train-One' &&
        targetTokenIndex !== null &&
        ffnOutput.length > 0) ||
      (trainingMode === 'Train-All' &&
        isTrainAllRunning &&
        ffnOutput.length > 1);

    if (shouldRunTraining) {
      // Start a timer that updates every second
      timerId = window.setInterval(() => {
        // Train-All mode: train on entire sequence
        if (
          trainingMode === 'Train-All' &&
          isTrainAllRunning &&
          ffnOutput.length > 1
        ) {
          // For Train-All, each token predicts the next token in the sequence
          const losses: number[] = [];
          const allGradients: number[][] = [];

          // Calculate loss and gradients for each position
          for (let pos = 0; pos < ffnOutput.length - 1; pos++) {
            const predictedEmbedding = ffnOutput[pos];
            const actualNextTokenIdx = selectedTokenIndices[pos + 1];
            const targetEmbedding = vocabularyEmbeddings[actualNextTokenIdx];

            // Check for invalid values
            if (hasInvalidValuesVector(predictedEmbedding)) {
              continue;
            }

            // Calculate softmax probabilities for this position
            const dotProducts = vocabularyEmbeddings.map((vocabEmbedding) =>
              vectorDotProduct(predictedEmbedding, vocabEmbedding)
            );

            const maxDotProduct = Math.max(...dotProducts);
            const expValues = dotProducts.map((dp) =>
              Math.exp(dp - maxDotProduct)
            );
            const sumExp = expValues.reduce((a, b) => a + b, 0);
            const probabilities = expValues.map((exp) => exp / sumExp);

            // Cross-entropy loss for this position
            const targetProb = probabilities[actualNextTokenIdx];
            const loss = -Math.log(Math.max(targetProb, 1e-7));
            losses.push(loss);

            // Gradient for this position
            const gradient = predictedEmbedding.map(
              (val, i) => val - targetEmbedding[i]
            );
            allGradients.push(gradient);
          }

          // Average loss across all positions
          const avgLoss =
            losses.length > 0
              ? losses.reduce((a, b) => a + b, 0) / losses.length
              : 0;
          setTrainingLoss(avgLoss);

          // Average gradients across all positions
          const avgGradient =
            allGradients.length > 0
              ? allGradients[0].map(
                  (_, i) =>
                    allGradients.reduce((sum, grad) => sum + grad[i], 0) /
                    allGradients.length
                )
              : Array(DIM_EMBEDDING).fill(0);

          // Apply gradient-based updates to all layers using averaged gradients
          // Update MLP weights with gradient descent
          setMlpWeights((prev) => {
            const newWeights = {
              W1: [...prev.W1.map((r) => [...r])],
              b1: [...prev.b1],
              W2: [...prev.W2.map((r) => [...r])],
              b2: [...prev.b2],
            };

            // Apply gradient updates to W2 (output layer)
            if (selectedElement?.matrixType === 'weightW2') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.W2.length; i++) {
                for (let j = 0; j < newWeights.W2[i].length; j++) {
                  if (i !== row || j !== col) {
                    const gradient = avgGradient[i] * 1.0;
                    newWeights.W2[i][j] -= learningRate * gradient;
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.W2.length; i++) {
                for (let j = 0; j < newWeights.W2[i].length; j++) {
                  const gradient = avgGradient[i] * 1.0;
                  newWeights.W2[i][j] -= learningRate * gradient;
                }
              }
            }

            // Apply gradient updates to biases
            for (let i = 0; i < newWeights.b2.length; i++) {
              newWeights.b2[i] -= learningRate * avgGradient[i] * 1.0;
            }

            // For W1 and b1, apply small gradient updates
            const avgError =
              avgGradient.reduce((a, b) => a + b, 0) / avgGradient.length;

            if (selectedElement?.matrixType === 'weightW1') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.W1.length; i++) {
                for (let j = 0; j < newWeights.W1[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.W1[i][j] -= learningRate * avgError * 0.5;
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.W1.length; i++) {
                for (let j = 0; j < newWeights.W1[i].length; j++) {
                  newWeights.W1[i][j] -= learningRate * avgError * 0.5;
                }
              }
            }

            // Update b1 biases
            for (let i = 0; i < newWeights.b1.length; i++) {
              newWeights.b1[i] -= learningRate * avgError * 0.5;
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

            const avgError =
              avgGradient.reduce((a, b) => a + b, 0) / avgGradient.length;
            const attentionLearningRate =
              learningRate * ATTENTION_LR_MULTIPLIER;

            // Update Q, K, V matrices
            ['weightQ', 'weightK', 'weightV'].forEach((matrixType) => {
              const matrix = matrixType as 'weightQ' | 'weightK' | 'weightV';
              const scaleFactor = matrix === 'weightV' ? 0.5 : 0.1;

              if (selectedElement?.matrixType === matrix) {
                const { row, col } = selectedElement;
                for (let i = 0; i < newWeights[matrix].length; i++) {
                  for (let j = 0; j < newWeights[matrix][i].length; j++) {
                    if (i !== row || j !== col) {
                      newWeights[matrix][i][j] -=
                        attentionLearningRate * avgError * scaleFactor;
                    }
                  }
                }
              } else {
                for (let i = 0; i < newWeights[matrix].length; i++) {
                  for (let j = 0; j < newWeights[matrix][i].length; j++) {
                    newWeights[matrix][i][j] -=
                      attentionLearningRate * avgError * scaleFactor;
                  }
                }
              }
            });

            return newWeights;
          });

          // Update training history for Train-All mode
          setTotalTrainingSteps((prev) => prev + 1);

          // For Train-All, we show average loss
          const historyItem: HistoryTrainingEntry = {
            loss: avgLoss,
            targetToken: 'all',
            predictedToken: 'all',
            targetProb: (1 / losses.length).toFixed(EXPONENTIAL_DECIMALS),
          };

          setHistoryTraining((prev: HistoryTrainingEntry[]) => {
            const newHistory = [...prev, historyItem];
            return newHistory.slice(-HISTORY_DISPLAY_STEPS);
          });
        } else if (
          trainingMode === 'Train-One' &&
          targetTokenIndex !== null &&
          ffnOutput.length > 0
        ) {
          // Get the predicted embedding (last token's output)
          const predictedEmbedding = ffnOutput[ffnOutput.length - 1];

          // Check for invalid values in the prediction
          if (hasInvalidValuesVector(predictedEmbedding)) {
            setCalculationError(
              'Invalid values detected in predicted embedding'
            );
            return;
          }

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

            // Use the multiplied learning rate for attention weights
            const attentionLearningRate =
              learningRate * ATTENTION_LR_MULTIPLIER;

            // Update Q matrix with gradient descent (skip selected element)
            if (selectedElement?.matrixType === 'weightQ') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightQ.length; i++) {
                for (let j = 0; j < newWeights.weightQ[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightQ[i][j] -=
                      attentionLearningRate * avgError * 0.1; // Increased from 0.001
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightQ.length; i++) {
                for (let j = 0; j < newWeights.weightQ[i].length; j++) {
                  newWeights.weightQ[i][j] -=
                    attentionLearningRate * avgError * 0.1; // Increased from 0.001
                }
              }
            }

            // Update K matrix with gradient descent (skip selected element)
            if (selectedElement?.matrixType === 'weightK') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightK.length; i++) {
                for (let j = 0; j < newWeights.weightK[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightK[i][j] -=
                      attentionLearningRate * avgError * 0.1; // Increased from 0.001
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightK.length; i++) {
                for (let j = 0; j < newWeights.weightK[i].length; j++) {
                  newWeights.weightK[i][j] -=
                    attentionLearningRate * avgError * 0.1; // Increased from 0.001
                }
              }
            }

            // Update V matrix with gradient descent - larger updates as it affects output directly
            if (selectedElement?.matrixType === 'weightV') {
              const { row, col } = selectedElement;
              for (let i = 0; i < newWeights.weightV.length; i++) {
                for (let j = 0; j < newWeights.weightV[i].length; j++) {
                  if (i !== row || j !== col) {
                    newWeights.weightV[i][j] -=
                      attentionLearningRate * avgError * 0.5; // Increased from 0.01
                  }
                }
              }
            } else {
              for (let i = 0; i < newWeights.weightV.length; i++) {
                for (let j = 0; j < newWeights.weightV[i].length; j++) {
                  newWeights.weightV[i][j] -=
                    attentionLearningRate * avgError * 0.5; // Increased from 0.01
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
    isTrainAllRunning,
    selectedTokenIndices,
    DIM_EMBEDDING,
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
      if (trainingMode === 'Train-One') {
        // In Train-One mode, set as target output
        setTargetTokenIndex(index);
      } else {
        // In Train-All or Inference mode, add to input sequence
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

        // First check if the new value is valid
        if (!isFinite(newValue) || isNaN(newValue)) {
          setCalculationError(`Invalid value entered: ${newValue}`);
          return;
        }

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

            // Check if the updated embeddings have any invalid values
            if (hasInvalidValues(newVocabEmbeddings)) {
              setCalculationError(
                'Invalid values detected after embedding update'
              );
              return;
            }

            setVocabularyEmbeddings(newVocabEmbeddings);
            // Clear any existing errors since the update was successful
            setCalculationError(null);
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
    <div className="min-h-screen bg-gray-50 overflow-x-hidden">
      <main className="w-full p-0.5 md:p-2 max-w-full">
        <div className="bg-white rounded p-0.5 mb-0.5">
          {/* Error banner */}
          {calculationError && (
            <div className="bg-red-600 text-white px-4 py-2 rounded-t-lg flex justify-between items-center">
              <div className="flex items-center gap-2">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <span className="font-medium">
                  Calculation Error: {calculationError}
                </span>
              </div>
              <button
                onClick={() => setCalculationError(null)}
                className="text-white hover:text-gray-200 focus:outline-none"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          )}

          {/* Main control panel */}
          <div
            className={`mb-4 ${
              trainingMode === 'Inferencing'
                ? 'bg-gradient-to-r from-blue-50 to-indigo-50'
                : trainingMode === 'Train-One'
                ? 'bg-gradient-to-r from-green-50 to-emerald-50'
                : 'bg-gradient-to-r from-purple-50 to-pink-50'
            } ${
              calculationError ? 'rounded-b-lg' : 'rounded-lg'
            } shadow-sm overflow-hidden`}
          >
            {/* Header bar */}
            <div
              className={`${
                trainingMode === 'Inferencing'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600'
                  : trainingMode === 'Train-One'
                  ? 'bg-gradient-to-r from-green-500 to-emerald-600'
                  : 'bg-gradient-to-r from-purple-500 to-pink-600'
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
                  onClick={() => {
                    // Cycle through modes: Inferencing -> Train-One -> Train-All -> Inferencing
                    const nextMode: Record<TrainingMode, TrainingMode> = {
                      Inferencing: 'Train-One',
                      'Train-One': 'Train-All',
                      'Train-All': 'Inferencing',
                    };
                    setTrainingMode(nextMode[trainingMode]);
                    // Clear target token when switching modes
                    setTargetTokenIndex(null);
                  }}
                  className={`px-4 sm:px-6 h-8 flex items-center justify-center text-xs sm:text-sm font-medium transition-all duration-200 rounded-lg shadow-sm border ${
                    trainingMode === 'Inferencing'
                      ? 'bg-blue-500 hover:bg-blue-600 text-white border-blue-600'
                      : trainingMode === 'Train-One'
                      ? 'bg-green-500 hover:bg-green-600 text-white border-green-600'
                      : 'bg-purple-500 hover:bg-purple-600 text-white border-purple-600'
                  }`}
                >
                  <span className="block">{trainingMode}</span>
                </button>
              </div>
              {/* Training controls */}
              {trainingMode !== 'Inferencing' && (
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
                  {/* {targetTokenIndex === null && (
                    <span className="text-xs sm:text-sm text-gray-500 italic">
                      {trainingMode === 'Train-One' &&
                        'Click a token to set next token target'}
                      {trainingMode === 'Train-All' &&
                        'Click a token to set sequence target'}
                    </span>
                  )} */}
                </div>
              )}
            </div>
          </div>

          {/* Tokenizer section */}
          <div className="mb-0.5 bg-white rounded p-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Token Dictionary
            </h3>
            <div className="p-1 sm:p-2">
              <p className="text-[10px] sm:text-xs text-gray-600 mb-1 sm:mb-2">
                {trainingMode === 'Inferencing' && (
                  <>
                    Click tokens to add to input sequence (hover to see
                    embeddings)
                  </>
                )}
                {trainingMode === 'Train-One' && (
                  <>
                    Click tokens to set as target for next token prediction
                    (hover to see embeddings)
                  </>
                )}
                {trainingMode === 'Train-All' && (
                  <>
                    Click tokens to add to input sequence (hover to see
                    embeddings)
                  </>
                )}
              </p>
              <div className="flex flex-wrap gap-1 sm:gap-2">
                {vocabularyWords.map((word, idx) => (
                  <Token
                    key={idx}
                    text={word}
                    onClick={() => handleTokenizerClick(idx)}
                    tokenType="tokenizer"
                    isTargetToken={idx === targetTokenIndex}
                    isPredictedToken={false}
                    isRecentlyAdded={idx === recentlyAddedIndex}
                    isTrainingMode={trainingMode !== 'Inferencing'}
                    showEmbedding={true}
                    embedding={vocabularyEmbeddings[idx]}
                    embeddingDimension={DIM_EMBEDDING}
                  />
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
                      <Token
                        key={seqIdx}
                        text={vocabularyWords[tokenIdx]}
                        onClick={() => handleSequenceTokenClick(seqIdx)}
                        tokenType="input"
                        isTargetToken={false}
                        isTrainingMode={trainingMode !== 'Inferencing'}
                        showEmbedding={true}
                        embedding={vocabularyEmbeddings[tokenIdx]}
                        embeddingDimension={DIM_EMBEDDING}
                        includeHeight={true}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Output Token section */}
          <div className="mb-0.5 bg-white rounded p-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              {trainingMode === 'Train-All'
                ? 'Training Control'
                : 'Output Token'}
            </h3>
            <div className="p-1 sm:p-2">
              {trainingMode === 'Train-All' ? (
                // Train-All mode: Start/Stop button and training status
                <div className="flex flex-col items-center justify-center py-4">
                  <button
                    onClick={() => setIsTrainAllRunning(!isTrainAllRunning)}
                    className={`px-6 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${
                      isTrainAllRunning
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-purple-500 hover:bg-purple-600 text-white'
                    }`}
                  >
                    {isTrainAllRunning ? 'Stop Training' : 'Start Training'}
                  </button>
                  <p className="text-[10px] sm:text-xs text-gray-600 mt-3">
                    {isTrainAllRunning
                      ? 'Training entire sequence with masked attention...'
                      : 'Click to start training on the entire input sequence'}
                  </p>
                  {isTrainAllRunning && selectedTokenIndices.length > 1 && (
                    <div className="mt-4 text-[10px] sm:text-xs text-gray-700">
                      Training {selectedTokenIndices.length - 1} predictions
                      (tokens 1-{selectedTokenIndices.length - 1} predicting
                      tokens 2-{selectedTokenIndices.length})
                    </div>
                  )}
                </div>
              ) : (
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
                            vectorDotProduct(
                              nextTokenPrediction,
                              vocabEmbedding
                            )
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
                        const predictedProb =
                          probabilities[predictedTokenIndex];

                        return (
                          <div className="flex items-center gap-2">
                            <Token
                              text={vocabularyWords[predictedTokenIndex]}
                              tokenType="output_prediction"
                              showEmbedding={true}
                              embedding={
                                vocabularyEmbeddings[predictedTokenIndex]
                              }
                              embeddingDimension={DIM_EMBEDDING}
                            />
                            <span className="text-[10px] sm:text-xs text-gray-600 font-mono">
                              p: {predictedProb >= 0 ? '+' : ''}
                              {predictedProb.toExponential(
                                EXPONENTIAL_DECIMALS
                              )}
                            </span>
                          </div>
                        );
                      })()
                    ) : (
                      <Token text="computing..." tokenType="placeholder" />
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
                            vectorDotProduct(
                              nextTokenPrediction,
                              vocabEmbedding
                            )
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

                        // Find which columns match the target token (if in training mode)
                        const highlightColumns: number[] = [];
                        if (
                          trainingMode !== 'Inferencing' &&
                          targetTokenIndex !== null
                        ) {
                          const targetTokenName =
                            vocabularyWords[targetTokenIndex];
                          sortedTokenLabels.forEach((label, idx) => {
                            if (label === targetTokenName) {
                              highlightColumns.push(idx);
                            }
                          });
                        }

                        return (
                          <div className="overflow-x-auto">
                            <MatrixDisplay
                              data={[sortedSoftmax.map((item) => item.value)]}
                              rowLabels={undefined}
                              columnLabels={sortedTokenLabels}
                              maxAbsValue={1.0}
                              cellSize="xs"
                              selectable={false}
                              matrixType="none"
                              highlightColumns={highlightColumns}
                            />
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Target Output (Training Mode Only) */}
                  {trainingMode !== 'Inferencing' && (
                    <div className="flex-1">
                      <p className="text-[10px] sm:text-xs text-gray-600 mb-1">
                        Desired Next Token
                        {targetTokenIndex !== null && ' (click to remove)'}:
                      </p>
                      <div className="min-h-[40px] sm:min-h-[50px] border-2 border-dashed rounded-lg p-1 sm:p-2 transition-colors border-gray-300 bg-gray-50">
                        {targetTokenIndex !== null ? (
                          <div className="flex items-center gap-2">
                            <Token
                              text={vocabularyWords[targetTokenIndex]}
                              tokenType="output_target"
                              showEmbedding={true}
                              embedding={vocabularyEmbeddings[targetTokenIndex]}
                              embeddingDimension={DIM_EMBEDDING}
                              onClick={() => setTargetTokenIndex(null)}
                            />
                            {trainingLoss !== null && (
                              <span className="text-[10px] sm:text-xs text-gray-600 font-mono">
                                Loss: {trainingLoss >= 0 ? '+' : ''}
                                {trainingLoss.toExponential(
                                  EXPONENTIAL_DECIMALS
                                )}
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
              )}

              {/* Training Status Message */}
              {trainingMode === 'Train-One' && targetTokenIndex === null && (
                <p className="text-[10px] sm:text-xs text-gray-500 italic mt-2">
                  Click a token in the tokenizer to set as target output
                </p>
              )}
            </div>
          </div>

          {/* History Graphs */}
          {trainingMode === 'Train-One' && (
            <div className="flex flex-col lg:flex-row lg:gap-4">
              <div className="lg:flex-1">
                <HistoryGraphSoftmax
                  history={historySoftMax}
                  maxPoints={HISTORY_DISPLAY_STEPS}
                  vocabularyWords={vocabularyWords}
                  totalSteps={totalTrainingSteps}
                />
              </div>
              <div className="lg:flex-1">
                <HistoryGraphLoss
                  history={historyTraining}
                  maxPoints={HISTORY_DISPLAY_STEPS}
                  totalSteps={totalTrainingSteps}
                />
              </div>
            </div>
          )}

          {/* Train-All History Graph */}
          {trainingMode === 'Train-All' && (
            <div className="mb-0.5 bg-white rounded p-0.5">
              <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
                Training Progress
              </h3>
              <div className="p-2">
                <HistoryGraphLoss
                  history={historyTraining}
                  maxPoints={HISTORY_DISPLAY_STEPS}
                  totalSteps={totalTrainingSteps}
                />
                {isTrainAllRunning && selectedTokenIndices.length > 1 && (
                  <div className="mt-2 text-[10px] sm:text-xs text-gray-600">
                    Training {selectedTokenIndices.length - 1} token predictions
                    in parallel. Loss shown is the average across all positions.
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="mb-0.5">
            <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
              Embeddings with Positional Encoding
            </h3>
            <div
              className={`flex ${
                isMobile ? 'flex-col' : 'grid grid-cols-3'
              } lg:grid-cols-3 xl:grid-cols-12 gap-2 sm:gap-4 lg:gap-1 max-w-full`}
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
                <div className="w-full overflow-x-auto pb-2 max-w-full">
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
                    isTrainingMode={trainingMode !== 'Inferencing'}
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
                <div className="w-full overflow-x-auto pb-2 max-w-full">
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
                <div className="w-full overflow-x-auto pb-2 max-w-full">
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
              Self-Attention {trainingMode === 'Train-All' && '(Masked)'}
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
              onCalculationError={setCalculationError}
              useMaskedAttention={trainingMode === 'Train-All'}
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
                onCalculationError={setCalculationError}
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
        {ffnOutput.length > 0 &&
          (trainingMode !== 'Train-All' || !isTrainAllRunning) && (
            <div className="mt-0.5 mb-0.5">
              <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
                Next Token Prediction
              </h3>
              <div className="bg-white rounded p-0.5">
                <div
                  className={`flex ${
                    isMobile ? 'flex-col' : 'grid grid-cols-3'
                  } lg:grid-cols-3 xl:grid-cols-12 gap-2 sm:gap-3 lg:gap-1 max-w-full`}
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
                    const indexedSoftmax = softmaxValues.map(
                      (value, index) => ({
                        index,
                        value,
                      })
                    );
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
                            rowLabels={undefined}
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
                                  rowLabels={undefined}
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
                                  data={[
                                    sortedSoftmax.map((item) => item.value),
                                  ]}
                                  rowLabels={undefined}
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
                              <Token
                                text={topPredictedToken}
                                tokenType="output_prediction"
                                showEmbedding={true}
                                embedding={topPredictedTokenEmbedding}
                                embeddingDimension={DIM_EMBEDDING}
                              />
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
