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
    throw new Error(`Matrix dimensions don't match for multiplication: ${a[0].length} != ${b.length}`);
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
  return a.map(row => row.map(fn));
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
    throw new Error(`Dimensions don't match for bias addition: ${a[0].length} != ${b.length}`);
  }
  
  return a.map(row => row.map((val, i) => val + b[i]));
}

/**
 * ReLU activation function
 * @param x - Input value
 * @returns max(0, x)
 */
export function relu(x: number): number {
  return Math.max(0, x);
}

/**
 * Apply softmax to each row of a matrix
 * @param matrix - Input matrix
 * @returns Matrix with softmax applied to each row
 */
export function softmax(matrix: number[][]): number[][] {
  return matrix.map(row => {
    // Find the maximum value for numerical stability
    const max = Math.max(...row);
    
    // Calculate exp(x - max) for each element
    const expValues = row.map(val => Math.exp(val - max));
    
    // Sum of all exp values
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    
    // Normalize by dividing each by the sum
    return expValues.map(exp => exp / sumExp);
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
  return a.map(row => row.map(val => val * scalar));
}

/**
 * Generate a random matrix of specified shape
 * @param rows - Number of rows
 * @param cols - Number of columns
 * @param min - Minimum value (default: -1)
 * @param max - Maximum value (default: 1)
 * @returns Random matrix
 */
export function randomMatrix(rows: number, cols: number, min = -1, max = 1): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < rows; i++) {
    matrix[i] = [];
    for (let j = 0; j < cols; j++) {
      matrix[i][j] = min + Math.random() * (max - min);
    }
  }
  return matrix;
}

/**
 * Generate a random vector of specified length
 * @param size - Length of vector
 * @param min - Minimum value (default: -1)
 * @param max - Maximum value (default: 1)
 * @returns Random vector
 */
export function randomVector(size: number, min = -1, max = 1): number[] {
  return Array.from({ length: size }, () => min + Math.random() * (max - min));
}

/**
 * Generate sample input embeddings (one embedding per token)
 * @param numTokens - Number of tokens
 * @param embeddingDim - Embedding dimension
 * @returns Matrix where each row is a token embedding
 */
export function generateSampleEmbeddings(numTokens: number, embeddingDim: number): number[][] {
  return randomMatrix(numTokens, embeddingDim, -1, 1);
}

/**
 * Generate sample attention weights for Q, K, V projections
 * @param embeddingDim - Embedding dimension
 * @param headDim - Attention head dimension
 * @returns Object containing Q, K, V weight matrices
 */
export function generateSampleAttentionWeights(embeddingDim: number, headDim: number) {
  return {
    weightQ: randomMatrix(embeddingDim, headDim, -0.5, 0.5),
    weightK: randomMatrix(embeddingDim, headDim, -0.5, 0.5),
    weightV: randomMatrix(embeddingDim, headDim, -0.5, 0.5),
  };
}

/**
 * Generate sample MLP weights
 * @param inputDim - Input dimension
 * @param hiddenDim - Hidden layer dimension
 * @returns Object containing weights and biases
 */
export function generateSampleMLPWeights(inputDim: number, hiddenDim: number) {
  return {
    W1: randomMatrix(inputDim, hiddenDim, -0.5, 0.5),
    b1: randomVector(hiddenDim, -0.1, 0.1),
    W2: randomMatrix(hiddenDim, inputDim, -0.5, 0.5),
    b2: randomVector(inputDim, -0.1, 0.1),
  };
}