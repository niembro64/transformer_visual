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

  // Apply dropout to embeddings (only during training)
  const embeddingsWithDropout = useMemo(
    () => applyDropout(embeddings, embeddingDropoutRate, trainingMode),
    [embeddings, embeddingDropoutRate, trainingMode]
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
    // Initialize selectedElement to the first element of the first matrix
    if (rawEmbeddings.length > 0 && rawEmbeddings[0].length > 0) {
      return {
        matrixType: 'embeddings',
        row: 0,
        col: 0,
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
          {/* Combined top control bar with tokens and dimensions */}
          <div className="flex flex-col mb-4 border-b pb-4">
            <div className="flex justify-between items-start mb-3">
              {/* Left: Token controls - takes 2/3 of space */}
              <div className="w-2/3 pr-4">
                <h3 className="text-sm font-semibold mb-2">Edit Tokens</h3>
                <div className="flex flex-wrap gap-2">
                  {tokenLabels.map((token, index) => (
                    <div key={index} className="relative group">
                      {/* Delete button appears on hover */}
                      <button
                        className="absolute -top-2.5 -right-2.5 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-sm font-medium shadow-sm opacity-0 group-hover:opacity-100 transition-opacity"
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
                        className="px-2 py-1 border rounded text-sm min-w-[3rem] text-center"
                        placeholder="Token"
                      />
                    </div>
                  ))}

                  {/* Add token button */}
                  <button
                    className="px-2 py-1 border rounded bg-gray-50 hover:bg-gray-100 text-gray-500"
                    onClick={addToken}
                    title="Add token"
                  >
                    +
                  </button>
                </div>
              </div>

              {/* Right: Settings controls - takes 1/3 of space */}
              <div className="w-1/3 flex flex-col gap-3 pl-4 border-l">
                {/* Embedding dimension controls */}
                <div>
                  <h3 className="text-sm font-semibold mb-2">
                    Embedding Dimension
                  </h3>
                  <div className="flex items-center border rounded overflow-hidden w-32 mx-auto">
                    <button
                      className="px-2 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm"
                      onClick={decreaseEmbeddingDim}
                      disabled={embeddingDim <= 2}
                    >
                      -
                    </button>
                    <div className="px-4 py-1 flex-grow text-center font-medium">
                      {embeddingDim}
                    </div>
                    <button
                      className="px-2 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm"
                      onClick={increaseEmbeddingDim}
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Training mode toggle */}
                <div>
                  <h3 className="text-sm font-semibold mb-2">Training Mode</h3>
                  <div className="flex items-center justify-center">
                    <button
                      onClick={() => setTrainingMode(false)}
                      className={`px-4 py-1 border border-r-0 rounded-l transition-colors ${
                        !trainingMode
                          ? 'bg-gray-200 font-medium'
                          : 'bg-white text-gray-500'
                      }`}
                    >
                      Off
                    </button>
                    <button
                      onClick={() => setTrainingMode(true)}
                      className={`px-4 py-1 border border-l-0 rounded-r transition-colors ${
                        trainingMode
                          ? 'bg-blue-500 text-white font-medium'
                          : 'bg-white text-gray-500'
                      }`}
                    >
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
