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

function App() {
  // Configuration for the demo with new dimensions
  // const numTokens = 6; // Number of tokens in the sequence (6 tokens)
  const embeddingDim = 10; // Dimension of token embeddings (d_model = 4)
  const attentionHeadDim = 4; // Dimension of attention head (d_k/d_v = 3)
  const mlpHiddenDim = 8; // Dimension of MLP hidden layer (d_ff = 8, typically 4x d_model)
  // const maxSeqLength = 32;    // Maximum sequence length for positional encodings

  // Dropout rates
  const embeddingDropoutRate = 0.0; // Dropout rate after embeddings + positional encodings
  const attentionDropoutRate = 0.0; // Dropout rate after attention
  const ffnDropoutRate = 0.0; // Dropout rate in feed-forward network

  // Training mode - determines if dropout is applied
  const [trainingMode, setTrainingMode] = useState(false);

  // Token labels for 6 tokens - a simple sentence
  const tokenLabels: string[] = [
    'The',
    'cat',
    'sat',
    'on',
    'the',
    'mat',
    'hello',
    'world',
    'this',
    'is',
    'a',
    'test',
    // 'example',
    // 'sequence',
    // 'for',
    // 'attention',
    // 'and',
    // 'feed-forward',
    // 'networks',
    // 'in',
    // 'transformers',
    // 'with',
    // 'positional',
    // 'encoding',
    // 'and',
    // 'dropout',
  ];
  const maxSeqLength = tokenLabels.length; // Maximum sequence length for positional encodings

  // Generate positional encodings
  const [positionalEncodings] = useState(() =>
    generatePositionalEncodings(maxSeqLength, embeddingDim)
  );

  // Generate raw embeddings
  const [rawEmbeddings, setRawEmbeddings] = useState(() =>
    generateSampleEmbeddings(tokenLabels.length, embeddingDim)
  );

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

  // Sample data generation for attention weights
  const [attentionWeights, setAttentionWeights] = useState(() =>
    generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
  );

  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights, setMlpWeights] = useState(() =>
    generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
  );

  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

  // Only have one selectedElement for the entire application to prevent multiple sliders
  const [selectedElement, setSelectedElement] = useState<{
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
  } | null>(null);

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
          // Create a new copy of embeddings with the updated value
          const newRawEmbeddings = rawEmbeddings.map((r, i) =>
            i === row ? r.map((v, j) => (j === col ? newValue : v)) : [...r]
          );
          setRawEmbeddings(newRawEmbeddings);
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
          if (matrixType === 'weightQ') {
            newAttentionWeights.weightQ[row][col] = newValue;
          } else if (matrixType === 'weightK') {
            newAttentionWeights.weightK[row][col] = newValue;
          } else if (matrixType === 'weightV') {
            newAttentionWeights.weightV[row][col] = newValue;
          }

          setAttentionWeights(newAttentionWeights);
        } else if (matrixType === 'weightW1' || matrixType === 'weightW2') {
          // Create a deep copy of MLP weights
          const newMlpWeights = {
            W1: [...mlpWeights.W1.map((r) => [...r])],
            b1: [...mlpWeights.b1],
            W2: [...mlpWeights.W2.map((r) => [...r])],
            b2: [...mlpWeights.b2],
          };

          // Update the specific weight
          if (matrixType === 'weightW1') {
            newMlpWeights.W1[row][col] = newValue;
          } else if (matrixType === 'weightW2') {
            newMlpWeights.W2[row][col] = newValue;
          }

          setMlpWeights(newMlpWeights);
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
          {/* Training mode toggle */}
          <div className="mb-2 flex justify-end">
            <label className="inline-flex items-center cursor-pointer">
              <span className="text-[0.6rem] text-gray-700 mr-1">
                Training Mode
              </span>
              <div className="relative">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={trainingMode}
                  onChange={() => setTrainingMode(!trainingMode)}
                />
                <div className="w-8 h-4 bg-gray-200 peer-focus:outline-none rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </div>
            </label>
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
          <p className="text-gray-700">
            Blue: positive, Pink: negative. Click a value to edit (magenta
            border).
            {trainingMode && ' Training mode enabled: dropout is applied.'}
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
