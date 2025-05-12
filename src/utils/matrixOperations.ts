/**
 * Utility functions for matrix operations in the transformer visualization
 */

/**
 * Matrix multiplication: A x B
 * @param a - First matrix of shape [m, n]
 * @param b - Second matrix of shape [n, p]
 * @returns Result matrix of shape [m, p]
 */
export function matrixMultiply(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0) return [[]];
  if (a[0].length !== b.length) {
    throw new Error(
      `Matrix dimensions don't match for multiplication: ${a[0].length} != ${b.length}`
    );
  }

  const result: number[][] = [];
  const m = a.length;
  const n = a[0].length;
  const p = b[0].length;

  for (let i = 0; i < m; i++) {
    result[i] = [];
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}

/**
 * Matrix transpose
 * @param a - Matrix to transpose
 * @returns Transposed matrix
 */
export function transpose(a: number[][]): number[][] {
  if (a.length === 0) return [[]];

  const rows = a.length;
  const cols = a[0].length;
  const result: number[][] = [];

  for (let j = 0; j < cols; j++) {
    result[j] = [];
    for (let i = 0; i < rows; i++) {
      result[j][i] = a[i][j];
    }
  }

  return result;
}

/**
 * Apply a function element-wise to a matrix
 * @param a - Input matrix
 * @param fn - Function to apply to each element
 * @returns Transformed matrix
 */
export function applyFn(a: number[][], fn: (x: number) => number): number[][] {
  return a.map((row) => row.map(fn));
}

/**
 * Add a vector to each row of a matrix
 * @param a - Matrix
 * @param b - Vector to add to each row
 * @returns Result matrix
 */
export function addBias(a: number[][], b: number[]): number[][] {
  if (a.length === 0) return [[]];
  if (a[0].length !== b.length) {
    throw new Error(
      `Dimensions don't match for bias addition: ${a[0].length} != ${b.length}`
    );
  }

  return a.map((row) => row.map((val, i) => val + b[i]));
}

/**
 * Leaky ReLU activation function
 * @param x - Input value
 * @returns max(0.01*x, x) - allows small gradient when x < 0
 */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/**
 * Apply softmax to each row of a matrix
 * @param matrix - Input matrix
 * @returns Matrix with softmax applied to each row
 */
export function softmax(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    // Find the maximum value for numerical stability
    const max = Math.max(...row);

    // Calculate exp(x - max) for each element
    const expValues = row.map((val) => Math.exp(val - max));

    // Sum of all exp values
    const sumExp = expValues.reduce((a, b) => a + b, 0);

    // Normalize by dividing each by the sum
    return expValues.map((exp) => exp / sumExp);
  });
}

/**
 * Dot product of two matrices
 * @param a - First matrix
 * @param b - Second matrix
 * @returns Dot product matrix
 */
export function dotProduct(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0) return [[]];
  if (a.length !== b.length || a[0].length !== b[0].length) {
    throw new Error(`Matrix dimensions don't match for dot product`);
  }

  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] * b[i][j];
    }
  }

  return result;
}

/**
 * Scale a matrix by a scalar value
 * @param a - Matrix to scale
 * @param scalar - Scalar value
 * @returns Scaled matrix
 */
export function scaleMatrix(a: number[][], scalar: number): number[][] {
  return a.map((row) => row.map((val) => val * scalar));
}

/**
 * Generate a random value from normal distribution using Box-Muller transform
 * @param mean - Mean of the distribution (default: 0)
 * @param stdDev - Standard deviation (default: 1)
 * @returns Random value from normal distribution
 */
export function randomNormal(mean = 0, stdDev = 1): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

/**
 * Generate a matrix with values sampled from a normal distribution,
 * using typical initializations for neural networks
 * @param rows - Number of rows
 * @param cols - Number of columns
 * @param initMethod - Initialization method ('xavier', 'he', or 'scaled')
 * @returns Random matrix with values in typical neural network ranges
 */
export function randomNeuralMatrix(
  rows: number,
  cols: number,
  initMethod: 'xavier' | 'he' | 'scaled' = 'xavier'
): number[][] {
  const matrix: number[][] = [];

  let stdDev = 0.01; // Default small value

  // Calculate standard deviation based on initialization method
  if (initMethod === 'xavier') {
    // Xavier/Glorot initialization: good for tanh
    stdDev = Math.sqrt(2.0 / (rows + cols));
  } else if (initMethod === 'he') {
    // He initialization with increased scale for visualization
    stdDev = Math.sqrt(2.0 / rows) * 2.0; // Multiply by 2 to make effects more visible
  } else if (initMethod === 'scaled') {
    // Scaled initialization: good for attention
    stdDev = 1.0 / Math.sqrt(cols);
  }

  for (let i = 0; i < rows; i++) {
    matrix[i] = [];
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = randomNormal(0, stdDev);
    }
  }

  return matrix;
}

/**
 * Generate a vector with values sampled from a normal distribution,
 * typically used for bias terms in neural networks
 * @param size - Length of vector
 * @param stdDev - Standard deviation (default: 0.01)
 * @returns Random vector with values in typical neural network ranges
 */
export function randomNeuralVector(size: number, stdDev = 0.01): number[] {
  return Array.from({ length: size }, () => randomNormal(0, stdDev));
}

/**
 * Generate sample input embeddings (one embedding per token)
 * using realistic values from a trained model
 * @param numTokens - Number of tokens
 * @param embeddingDim - Embedding dimension
 * @returns Matrix where each row is a token embedding
 */
export function generateSampleEmbeddings(
  numTokens: number,
  embeddingDim: number
): number[][] {
  // Embeddings are typically drawn from a normal distribution with small standard deviation
  return randomNeuralMatrix(numTokens, embeddingDim, 'scaled');
}

/**
 * Generate sample attention weights for Q, K, V projections
 * using realistic values from a trained model
 * @param embeddingDim - Embedding dimension
 * @param headDim - Attention head dimension
 * @returns Object containing Q, K, V weight matrices
 */
export function generateSampleAttentionWeights(
  embeddingDim: number,
  headDim: number
) {
  // Attention weights are typically initialized with scaled initialization
  return {
    weightQ: randomNeuralMatrix(embeddingDim, headDim, 'scaled'),
    weightK: randomNeuralMatrix(embeddingDim, headDim, 'scaled'),
    weightV: randomNeuralMatrix(embeddingDim, headDim, 'scaled'),
  };
}

/**
 * Generate sample MLP weights for the feed-forward network
 * using realistic values from a trained model
 * @param inputDim - Input dimension
 * @param hiddenDim - Hidden layer dimension
 * @param attentionHeadDim - Attention head dimension (for input compatibility)
 * @returns Object containing weights and biases
 */
export function generateSampleMLPWeights(
  inputDim: number,
  hiddenDim: number,
  attentionHeadDim?: number
) {
  // If attentionHeadDim is provided, use that as the input dimension
  // This ensures compatibility when the input comes from attention output
  const actualInputDim = attentionHeadDim || inputDim;

  return {
    // Feed-forward weights typically use He initialization because of ReLU
    W1: randomNeuralMatrix(actualInputDim, hiddenDim, 'he'),
    b1: randomNeuralVector(hiddenDim),
    W2: randomNeuralMatrix(hiddenDim, inputDim, 'he'),
    b2: randomNeuralVector(inputDim),
  };
}
