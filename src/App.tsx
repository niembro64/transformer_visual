import React, { useState, useEffect, useCallback, useMemo } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import MatrixDisplay from './components/MatrixDisplay';
import {
  generateSampleEmbeddings,
  generateSampleAttentionWeights,
  generateSampleMLPWeights,
  generatePositionalEncodings,
  addPositionalEncodings,
  applyDropout,
  applyRandomWalk,
  applyRandomWalkToVector,
  relu,
} from './utils/matrixOperations';

export const dropoutUniveral = 0.03;

function App() {
  // Configurable dimension values
  const [embeddingDim, setEmbeddingDim] = useState(6); // Dimension of token embeddings (d_model)
  const attentionHeadDim = 2; // Dimension of attention head (d_k/d_v)
  const mlpHiddenDim = 4; // Dimension of MLP hidden layer (d_ff = 8, typically 4x d_model)

  // Dropout rates
  const embeddingDropoutRate = dropoutUniveral; // Dropout rate after embeddings + positional encodings
  const attentionDropoutRate = dropoutUniveral; // Dropout rate after attention
  const ffnDropoutRate = dropoutUniveral; // Dropout rate in feed-forward network

  // Training mode - determines if dropout is applied
  const [trainingMode, setTrainingMode] = useState(false);

  // Editable token list
  const [tokenLabels, setTokenLabels] = useState<string[]>([
    'a',
    'cat',
    'sat',
    'on',
    'the',
    'mat',
  ]);

  // Maximum sequence length - based on current token count
  const maxSeqLength = useMemo(
    () => tokenLabels.length * 2,
    [tokenLabels.length]
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

  // Generate raw embeddings - regenerate when tokens or dimensions change
  const [rawEmbeddings, setRawEmbeddings] = useState(() =>
    generateSampleEmbeddings(tokenLabels.length, embeddingDim)
  );

  // Regenerate embeddings when token count or dimensions change
  useEffect(() => {
    setRawEmbeddings(
      generateSampleEmbeddings(tokenLabels.length, embeddingDim)
    );
  }, [tokenLabels.length, embeddingDim]);

  // Apply positional encodings to embeddings
  const embeddings = useMemo(
    () => addPositionalEncodings(rawEmbeddings, positionalEncodings),
    [rawEmbeddings, positionalEncodings]
  );

  // For forcing re-renders on dropout timer cycles and weight updates
  const [trainingCycle, setTrainingCycle] = useState(0);

  // Used for random walk step size - smaller values for more subtle updates
  const weightUpdateStepSize = 0.003;

  // Timer to update dropout masks and perform weight random walks every second in training mode
  useEffect(() => {
    let timerId: number | null = null;

    if (trainingMode) {
      // Start a timer that updates every second
      timerId = window.setInterval(() => {
        setTrainingCycle((prev) => prev + 1); // Increment counter to trigger re-renders

        // Apply random walks to all trainable weights when in training mode
        if (trainingMode) {
          // Update raw embeddings (token embeddings)
          setRawEmbeddings((prev) =>
            applyRandomWalk(prev, weightUpdateStepSize, 'embeddings_weights')
          );

          // Update attention weights (Q, K, V projection matrices)
          setAttentionWeights((prev) => ({
            weightQ: applyRandomWalk(
              prev.weightQ,
              weightUpdateStepSize,
              'weightQ'
            ),
            weightK: applyRandomWalk(
              prev.weightK,
              weightUpdateStepSize,
              'weightK'
            ),
            weightV: applyRandomWalk(
              prev.weightV,
              weightUpdateStepSize,
              'weightV'
            ),
          }));

          // Update MLP weights and biases
          setMlpWeights((prev) => ({
            W1: applyRandomWalk(prev.W1, weightUpdateStepSize, 'mlp_w1'),
            b1: applyRandomWalkToVector(
              prev.b1,
              weightUpdateStepSize,
              'mlp_b1'
            ),
            W2: applyRandomWalk(prev.W2, weightUpdateStepSize, 'mlp_w2'),
            b2: applyRandomWalkToVector(
              prev.b2,
              weightUpdateStepSize,
              'mlp_b2'
            ),
          }));
        }
      }, 1000);
    }

    return () => {
      if (timerId !== null) {
        window.clearInterval(timerId);
      }
    };
  }, [trainingMode, weightUpdateStepSize]);

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

  // Token manipulation functions
  const addToken = useCallback(() => {
    setTokenLabels((prevTokens) => [...prevTokens, '']);
  }, []);

  const removeToken = useCallback((index: number) => {
    setTokenLabels((prevTokens) => {
      const newTokens = [...prevTokens];
      newTokens.splice(index, 1);
      return newTokens;
    });
  }, []);

  const updateToken = useCallback((index: number, newText: string) => {
    setTokenLabels((prevTokens) => {
      const newTokens = [...prevTokens];
      newTokens[index] = newText;
      return newTokens;
    });
  }, []);

  // Embedding dimension adjustment
  const increaseEmbeddingDim = useCallback(() => {
    setEmbeddingDim((prev) => prev + 2); // Increase by 2 to keep it even for positional encodings
    setSelectedElement(null); // Reset selection when changing dimensions
  }, []);

  const decreaseEmbeddingDim = useCallback(() => {
    setEmbeddingDim((prev) => Math.max(2, prev - 2)); // Decrease by 2, but ensure minimum of 2
    setSelectedElement(null); // Reset selection when changing dimensions
  }, []);

  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

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

  const initialElement: ElementObject | null = useMemo(() => {
    // Initialize selectedElement to the second-to-last token's 4th embedding
    if (rawEmbeddings.length > 0 && rawEmbeddings[0].length > 0) {
      const tokenIndex = Math.max(0, rawEmbeddings.length - 2); // Second-to-last token
      const embIndex = Math.min(3, rawEmbeddings[0].length - 1); // 4th embedding (index 3) or last if fewer

      return {
        matrixType: 'embeddings',
        row: tokenIndex,
        col: embIndex,
      };
    }
    return null;
  }, [rawEmbeddings]);

  // Only have one selectedElement for the entire application to prevent multiple sliders
  const [selectedElement, setSelectedElement] = useState<ElementObject | null>(
    initialElement
  );

  // State for the current value of the selected element
  const [selectedValue, setSelectedValue] = useState<number | null>(null);

  // Update selectedValue when selectedElement changes
  useEffect(() => {
    if (selectedElement) {
      const { matrixType, row, col } = selectedElement;

      if (matrixType === 'embeddings') {
        setSelectedValue(rawEmbeddings[row][col]); // Show original embedding values before positional encoding
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
          // Only update if row and col are within bounds (they may not be if we've removed tokens or changed dim)
          if (row < rawEmbeddings.length && col < rawEmbeddings[0].length) {
            // Create a new copy of embeddings with the updated value
            const newRawEmbeddings = rawEmbeddings.map((r, i) =>
              i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
            );
            setRawEmbeddings(newRawEmbeddings);
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

  return (
    <div className="min-h-screen bg-gray-50">
      <main className="w-full p-0.5">
        <div className="bg-white rounded p-0.5 mb-0.5">
          {/* Main control panel */}
          <div className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg shadow-sm overflow-hidden">
            {/* Header bar */}
            <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-4 py-2 flex justify-between items-center">
              <h2 className="text-lg font-bold">Transformer Configuration</h2>
            </div>

            {/* Content area */}
            <div className="p-4 flex justify-between items-start">
              {/* Left: Token controls - takes 2/3 of space */}
              <div className="w-2/3 pr-5">
                <div className="bg-white rounded-md shadow-sm p-3">
                  <h3 className="text-sm font-bold mb-3 text-gray-800 border-b pb-2 flex items-center">
                    <svg
                      className="w-4 h-4 mr-1 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"
                      />
                    </svg>
                    Edit Tokens
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {tokenLabels.map((token, index) => (
                      <div key={index} className="relative group">
                        {/* Delete button appears on hover */}
                        <button
                          className="absolute -top-2.5 -right-2.5 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-sm font-medium shadow-sm opacity-0 group-hover:opacity-100 transition-opacity z-10"
                          onClick={() => removeToken(index)}
                          title="Remove token"
                        >
                          ×
                        </button>

                        {/* Editable token input */}
                        <input
                          type="text"
                          value={token}
                          onChange={(e) => updateToken(index, e.target.value)}
                          className="px-3 py-1.5 border rounded text-sm min-w-[3.5rem] h-[34px] text-center shadow-sm hover:border-blue-300 focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 transition-colors"
                          placeholder="Token"
                        />
                      </div>
                    ))}

                    {/* Add token button */}
                    <button
                      className="px-3 py-1.5 border rounded min-w-[3.5rem] h-[34px] text-center bg-blue-50 hover:bg-blue-100 text-blue-600 transition-colors shadow-sm flex items-center justify-center"
                      onClick={addToken}
                      title="Add token"
                    >
                      +
                    </button>
                  </div>
                </div>
              </div>

              {/* Right: Settings controls - takes 1/3 of space */}
              <div className="w-1/3 flex flex-col gap-4">
                {/* Embedding dimension controls */}
                <div className="bg-white rounded-md shadow-sm p-3">
                  <h3 className="text-sm font-bold mb-3 text-gray-800 border-b pb-2 flex items-center">
                    <svg
                      className="w-4 h-4 mr-1 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
                      />
                    </svg>
                    Embedding Dimension
                  </h3>
                  <div className="flex items-center border rounded-lg overflow-hidden shadow-sm w-36 mx-auto">
                    <button
                      className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 text-sm font-medium transition-colors"
                      onClick={decreaseEmbeddingDim}
                      disabled={embeddingDim <= 2}
                    >
                      −
                    </button>
                    <div className="px-5 py-2 flex-grow text-center font-bold bg-white">
                      {embeddingDim}
                    </div>
                    <button
                      className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 text-sm font-medium transition-colors"
                      onClick={increaseEmbeddingDim}
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Training mode toggle */}
                <div className="bg-white rounded-md shadow-sm p-3">
                  <h3 className="text-sm font-bold mb-3 text-gray-800 border-b pb-2 flex items-center">
                    <svg
                      className="w-4 h-4 mr-1 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                      />
                    </svg>
                    Training Mode
                  </h3>

                  <div className="flex items-center justify-center w-36 mx-auto">
                    <button
                      onClick={() => setTrainingMode(false)}
                      className={`px-4 py-2 border shadow-sm font-medium rounded-l-md transition-colors ${
                        !trainingMode
                          ? 'bg-gray-200 border-gray-400 text-gray-800'
                          : 'bg-white text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      Off
                    </button>
                    <button
                      onClick={() => setTrainingMode(true)}
                      className={`px-4 py-2 border shadow-sm font-medium rounded-r-md transition-colors ${
                        trainingMode
                          ? 'bg-blue-500 border-blue-600 text-white'
                          : 'bg-white text-gray-500 hover:bg-gray-50'
                      } flex items-center`}
                    >
                      {trainingMode && (
                        <div className="mr-1 h-2 w-2 rounded-full bg-green-400 animate-pulse"></div>
                      )}
                      On
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-0.5">
            <h3 className="text-sm font-semibold mb-0.5 border-b pb-0.5">
              Embeddings with Positional Encoding
            </h3>
            <div className="grid grid-cols-12 gap-1">
              {/* Left: Raw Embeddings */}
              <div className="col-span-4 flex flex-col items-center">
                <h4 className="text-[0.65rem] font-medium mb-0.5">
                  Raw Token Embeddings
                </h4>
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
                  autoOscillate={true}
                />
              </div>

              {/* Middle: Positional Encodings */}
              <div className="col-span-4 flex flex-col items-center">
                <h4 className="text-[0.65rem] font-medium mb-0.5">
                  Positional Encodings
                </h4>
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

              {/* Right: Combined embeddings with positional encoding */}
              <div className="col-span-4 flex flex-col items-center">
                <h4 className="text-[0.65rem] font-medium mb-0.5">
                  Embeddings + Pos. Encoding
                  {trainingMode ? ` + Dropout(${embeddingDropoutRate})` : ''}
                </h4>
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

          <div className="mb-0.5">
            <h3 className="text-sm font-semibold mb-0.5 border-b pb-0.5">
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
                selectedElement={selectedElement}
                onElementClick={handleElementClick}
                onValueChange={handleValueChange}
                dropoutRate={ffnDropoutRate}
                applyTrainingDropout={trainingMode}
                activationFn={relu} // ReLU activation as default
                activationFnName="ReLU"
                dropoutCycle={trainingCycle} // Pass the training cycle to force updates
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
          <div className="flex flex-col gap-1">
            <p className="text-gray-700">
              Blue: positive, Red: negative. Click a value to edit (magenta
              border).
              {trainingMode && ' Training mode enabled: dropout is applied.'}
            </p>

            <div className="flex flex-wrap gap-4">
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
