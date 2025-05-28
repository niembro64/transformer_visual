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
 * Element-wise addition of two matrices
 * @param a - First matrix
 * @param b - Second matrix
 * @returns Result matrix
 */
export function matrixAdd(a: number[][], b: number[][]): number[][] {
  if (a.length === 0 || b.length === 0) return [[]];
  if (a.length !== b.length || a[0].length !== b[0].length) {
    throw new Error(
      `Matrix dimensions don't match for addition: [${a.length},${a[0].length}] != [${b.length},${b[0].length}]`
    );
  }

  const result: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    result[i] = [];
    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] + b[i][j];
    }
  }

  return result;
}

/**
 * ReLU activation function
 * @param x - Input value
 * @returns max(0, x) - zero when negative, identity when positive
 */
export function relu(x: number): number {
  return x > 0 ? x : 0;
}

/**
 * Leaky ReLU activation function
 * @param x - Input value
 * @param alpha - Slope for negative values (default: 0.01)
 * @returns x when positive, alpha*x when negative
 */
export function leakyRelu(x: number, alpha: number = 0.01): number {
  return x > 0 ? x : alpha * x;
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
 * Compute the dot product between two vectors (1D arrays)
 * @param a - First vector
 * @param b - Second vector
 * @returns Scalar dot product value
 */
export function vectorDotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimensions don't match for dot product: ${a.length} != ${b.length}`);
  }
  
  return a.reduce((sum, val, i) => sum + (val * b[i]), 0);
}

/**
 * Compute the cosine similarity between two vectors
 * @param a - First vector
 * @param b - Second vector
 * @returns Similarity value between -1 and 1
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = vectorDotProduct(a, b);
  
  // Calculate magnitudes (L2 norms)
  const magA = Math.sqrt(a.reduce((sum, val) => sum + (val * val), 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + (val * val), 0));
  
  // Avoid division by zero
  if (magA === 0 || magB === 0) return 0;
  
  return dotProduct / (magA * magB);
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
 * @param strengthMultiplier - Multiplier to adjust embedding strength (default 1.0)
 * @returns Matrix where each row is a token embedding
 */
export function generateSampleEmbeddings(
  numTokens: number,
  embeddingDim: number,
  strengthMultiplier: number = 1.0
): number[][] {
  // Create a pool of embeddings that are equidistant from each other
  const embeddingPool: number[][] = [];
  
  // We'll use a combination of orthogonal vectors and hypercube vertices
  // to ensure equal distances between embeddings
  
  // Step 1: Generate embeddings on the vertices of a hypercube
  // This ensures all embeddings are equidistant from the origin and from each other
  const poolSize = Math.max(numTokens * 2, Math.pow(2, Math.min(embeddingDim, 8)));
  
  // Base value for the hypercube (controls overall magnitude)
  const baseValue = 0.1 * strengthMultiplier;
  
  for (let i = 0; i < poolSize; i++) {
    const embedding = new Array(embeddingDim).fill(0);
    
    // Create a pattern that distributes positive and negative values
    // Use Gray code to ensure adjacent patterns differ by exactly one bit
    const grayCode = i ^ (i >> 1);
    
    for (let j = 0; j < embeddingDim; j++) {
      // Determine sign based on Gray code bit pattern
      const isPositive = (grayCode >> (j % 32)) & 1;
      
      // Create a base value that alternates between positive and negative
      let value = isPositive ? baseValue : -baseValue;
      
      // Add a dimension-specific scaling to create more variety
      // Use prime numbers to avoid repetitive patterns
      const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
      const prime = primes[j % primes.length];
      const scale = 1 + 0.3 * Math.sin(i * prime * 0.1);
      
      value *= scale;
      
      // Add a small rotation to prevent exact alignment
      // This maintains equal distances while adding variety
      const rotation = 0.1 * Math.cos((i + j) * 0.5);
      value += rotation * baseValue * 0.2;
      
      embedding[j] = value;
    }
    
    // Normalize to ensure all embeddings have the same magnitude
    // This maintains equal distances from the origin
    let magnitude = 0;
    for (let j = 0; j < embeddingDim; j++) {
      magnitude += embedding[j] * embedding[j];
    }
    magnitude = Math.sqrt(magnitude);
    
    // Target magnitude that ensures good separation
    const targetMagnitude = Math.sqrt(embeddingDim) * baseValue;
    if (magnitude > 0) {
      for (let j = 0; j < embeddingDim; j++) {
        embedding[j] = (embedding[j] / magnitude) * targetMagnitude;
      }
    }
    
    embeddingPool.push(embedding);
  }
  
  // Phase 2: Create a deterministic pseudo-random mapping from tokens to embeddings
  // Use a simple hash function based on token index and embedding dimension
  const tokenEmbeddings: number[][] = [];
  const usedIndices = new Set<number>();
  
  for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
    // Create a deterministic "random" index based on token index and embedding dimension
    // This ensures the same token always gets the same embedding for given parameters
    let hash = tokenIdx * 2654435761; // Large prime for better distribution
    hash = hash ^ (embeddingDim * 1597); // Mix in embedding dimension
    hash = (hash ^ (hash >>> 16)) * 0x85ebca6b;
    hash = (hash ^ (hash >>> 13)) * 0xc2b2ae35;
    hash = hash ^ (hash >>> 16);
    
    // Map hash to valid pool index
    let poolIndex = Math.abs(hash) % embeddingPool.length;
    
    // If this embedding is already used, find the next available one
    let attempts = 0;
    while (usedIndices.has(poolIndex) && attempts < embeddingPool.length) {
      poolIndex = (poolIndex + 1) % embeddingPool.length;
      attempts++;
    }
    
    // Mark as used and assign
    usedIndices.add(poolIndex);
    tokenEmbeddings.push([...embeddingPool[poolIndex]]); // Deep copy
  }
  
  return tokenEmbeddings;
}

/**
 * Generate sinusoidal positional encodings as described in the "Attention Is All You Need" paper
 * @param maxSeqLength - Maximum sequence length
 * @param embeddingDim - Embedding dimension (must be even)
 * @returns Matrix where each row is a positional encoding vector for a position
 */
export function generatePositionalEncodings(
  maxSeqLength: number,
  embeddingDim: number
): number[][] {
  if (embeddingDim % 2 !== 0) {
    throw new Error(
      'Embedding dimension must be even for sinusoidal positional encodings'
    );
  }

  const encodings: number[][] = [];

  for (let pos = 0; pos < maxSeqLength; pos++) {
    const row: number[] = new Array(embeddingDim).fill(0);

    for (let i = 0; i < embeddingDim; i += 2) {
      const angle = pos / Math.pow(10000, (2 * (i / 2)) / embeddingDim);
      row[i] = Math.sin(angle); // even
      row[i + 1] = Math.cos(angle); // odd
    }
    encodings.push(row);
  }
  return encodings;
}

/**
 * Add positional encodings to token embeddings
 * @param embeddings - Token embeddings matrix
 * @param posEncodings - Positional encodings matrix
 * @returns Matrix with positional information added to embeddings
 */
export function addPositionalEncodings(
  embeddings: number[][],
  posEncodings: number[][]
): number[][] {
  if (embeddings.length === 0) return embeddings;
  if (embeddings.length > posEncodings.length) {
    throw new Error(
      `Sequence length ${embeddings.length} exceeds maximum supported length ${posEncodings.length}`
    );
  }

  if (embeddings[0].length !== posEncodings[0].length) {
    throw new Error(
      `Embedding dimension ${embeddings[0].length} doesn't match positional encoding dimension ${posEncodings[0].length}`
    );
  }

  // Select just the positional encodings we need for our sequence length
  const neededPosEncodings = posEncodings.slice(0, embeddings.length);

  // Add positional encodings to token embeddings
  return matrixAdd(embeddings, neededPosEncodings);
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
  // Generate deterministic, distinct attention weights
  const scaleFactor = 1.0 / Math.sqrt(headDim); // Scaled initialization
  
  // Helper function to create systematic weight patterns
  const createSystematicMatrix = (rows: number, cols: number, seed: number): number[][] => {
    const matrix: number[][] = [];
    
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Create a deterministic pattern using trigonometric functions
        // This ensures smooth variations and both positive/negative values
        const angle1 = (i + seed) * Math.PI / rows;
        const angle2 = (j + seed * 2) * Math.PI / cols;
        
        // Combine multiple frequencies for richness
        let value = Math.sin(angle1) * Math.cos(angle2) * 0.7;
        value += Math.sin(angle1 * 2) * Math.sin(angle2 * 3) * 0.3;
        
        // Add a unique offset based on position and seed
        const offset = Math.cos((i * cols + j + seed * 10) * 0.1) * 0.2;
        value += offset;
        
        // Scale by initialization factor
        row.push(value * scaleFactor);
      }
      matrix.push(row);
    }
    
    return matrix;
  };
  
  return {
    weightQ: createSystematicMatrix(embeddingDim, headDim, 1), // Seed 1 for Q
    weightK: createSystematicMatrix(embeddingDim, headDim, 7), // Seed 7 for K  
    weightV: createSystematicMatrix(embeddingDim, headDim, 13), // Seed 13 for V
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

  // Generate deterministic, distinct MLP weights
  // He initialization scale factor for ReLU activation
  const heScaleW1 = Math.sqrt(2.0 / actualInputDim) * 2.0;
  const heScaleW2 = Math.sqrt(2.0 / hiddenDim) * 2.0;
  
  // Helper function to create systematic weight patterns
  const createSystematicMatrix = (rows: number, cols: number, scale: number, seed: number): number[][] => {
    const matrix: number[][] = [];
    
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Create distinct patterns using a combination of techniques
        // Use prime numbers to avoid repetitive patterns
        const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        const prime1 = primes[(i + seed) % primes.length];
        const prime2 = primes[(j + seed * 2) % primes.length];
        
        // Create wave pattern with varying frequencies
        const wave1 = Math.sin((i * prime1 + j) * 0.2 + seed);
        const wave2 = Math.cos((i + j * prime2) * 0.15 + seed * 1.5);
        
        // Combine waves with different weights
        let value = wave1 * 0.6 + wave2 * 0.4;
        
        // Add a diagonal gradient for additional variation
        const diagonal = ((i / rows) - (j / cols)) * 0.3;
        value += diagonal * Math.sin(seed * 0.7);
        
        // Scale by initialization factor
        row.push(value * scale);
      }
      matrix.push(row);
    }
    
    return matrix;
  };
  
  // Helper function to create systematic bias vectors
  const createSystematicVector = (size: number, seed: number): number[] => {
    const vector: number[] = [];
    const scale = 0.1; // Small bias initialization
    
    for (let i = 0; i < size; i++) {
      // Create a smooth pattern across the bias vector
      const angle = (i + seed * 3) * Math.PI / size;
      const value = Math.sin(angle) * Math.cos(angle * 2 + seed) * scale;
      vector.push(value);
    }
    
    return vector;
  };

  return {
    // Use different seeds for each weight matrix to ensure distinctness
    W1: createSystematicMatrix(actualInputDim, hiddenDim, heScaleW1, 31),
    b1: createSystematicVector(hiddenDim, 37),
    W2: createSystematicMatrix(hiddenDim, inputDim, heScaleW2, 41),
    b2: createSystematicVector(inputDim, 43),
  };
}

/**
 * Applies a small random walk to a matrix to simulate weight updates during training
 * @param matrix - The matrix to update
 * @param stepSize - The size of random walk steps (standard deviation of changes)
 * @param walkId - Optional identifier for this random walk instance (for caching)
 * @returns Updated matrix with small random changes
 */
export function applyRandomWalk(
  matrix: number[][],
  stepSize: number = 0.005,
  walkId: string = 'default'
): number[][] {
  if (matrix.length === 0 || matrix[0].length === 0) return matrix;
  
  // Create deep copy of the matrix to avoid mutating the original
  const result: number[][] = [];
  
  // Apply small random changes to each element
  for (let i = 0; i < matrix.length; i++) {
    result[i] = [];
    for (let j = 0; j < matrix[0].length; j++) {
      // Use randomNormal to get a change value with mean 0
      const change = randomNormal(0, stepSize);
      result[i][j] = matrix[i][j] + change;
    }
  }
  
  return result;
}

/**
 * Applies a small random walk to a vector to simulate bias updates during training
 * @param vector - The vector to update
 * @param stepSize - The size of random walk steps (standard deviation of changes)
 * @param walkId - Optional identifier for this random walk instance (for caching)
 * @returns Updated vector with small random changes
 */
export function applyRandomWalkToVector(
  vector: number[],
  stepSize: number = 0.005,
  walkId: string = 'default'
): number[] {
  if (vector.length === 0) return vector;
  
  // Create a copy of the vector to avoid mutating the original
  const result: number[] = [];
  
  // Apply small random changes to each element
  for (let i = 0; i < vector.length; i++) {
    // Use randomNormal to get a change value with mean 0
    const change = randomNormal(0, stepSize);
    result[i] = vector[i] + change;
  }
  
  return result;
}

/**
 * Utility function to check if device is in portrait orientation (height > width)
 * @returns boolean indicating if device is in portrait orientation
 */
export function isPortraitOrientation(): boolean {
  if (typeof window === 'undefined') return false; // Default to landscape for SSR
  return window.innerHeight > window.innerWidth;
}

/**
 * Calculate mean squared error loss between predicted and target
 * @param predicted - Predicted values
 * @param target - Target values
 * @returns MSE loss
 */
export function mseLoss(predicted: number[], target: number[]): number {
  if (predicted.length !== target.length) {
    throw new Error('Predicted and target must have same length');
  }
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / predicted.length;
}

/**
 * Calculate gradient for MSE loss
 * @param predicted - Predicted values
 * @param target - Target values
 * @returns Gradient vector
 */
export function mseGradient(predicted: number[], target: number[]): number[] {
  if (predicted.length !== target.length) {
    throw new Error('Predicted and target must have same length');
  }
  return predicted.map((p, i) => 2 * (p - target[i]) / predicted.length);
}

/**
 * Update matrix weights using gradient descent
 * @param weights - Current weights
 * @param gradients - Gradients for each weight
 * @param learningRate - Learning rate
 * @returns Updated weights
 */
export function updateWeights(
  weights: number[][],
  gradients: number[][],
  learningRate: number
): number[][] {
  if (weights.length !== gradients.length || weights[0].length !== gradients[0].length) {
    throw new Error('Weights and gradients must have same dimensions');
  }
  return weights.map((row, i) =>
    row.map((w, j) => w - learningRate * gradients[i][j])
  );
}

/**
 * Update vector weights using gradient descent
 * @param weights - Current weights
 * @param gradients - Gradients for each weight
 * @param learningRate - Learning rate
 * @returns Updated weights
 */
export function updateVectorWeights(
  weights: number[],
  gradients: number[],
  learningRate: number
): number[] {
  if (weights.length !== gradients.length) {
    throw new Error('Weights and gradients must have same length');
  }
  return weights.map((w, i) => w - learningRate * gradients[i]);
}

/**
 * Check if a value is valid (not NaN, not Infinity)
 * @param value - Value to check
 * @returns true if value is valid, false otherwise
 */
export function isValidNumber(value: number): boolean {
  return !isNaN(value) && isFinite(value);
}

/**
 * Check if a matrix contains any invalid values (NaN or Infinity)
 * @param matrix - Matrix to check
 * @returns true if matrix contains invalid values, false otherwise
 */
export function hasInvalidValues(matrix: number[][]): boolean {
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (!isValidNumber(matrix[i][j])) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Check if a vector contains any invalid values (NaN or Infinity)
 * @param vector - Vector to check
 * @returns true if vector contains invalid values, false otherwise
 */
export function hasInvalidValuesVector(vector: number[]): boolean {
  return vector.some(val => !isValidNumber(val));
}

/**
 * Get error details for a matrix with invalid values
 * @param matrix - Matrix to check
 * @param matrixName - Name of the matrix for error reporting
 * @returns Error details or null if no errors
 */
export function getMatrixErrorDetails(matrix: number[][], matrixName: string): string | null {
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (!isValidNumber(matrix[i][j])) {
        const errorType = isNaN(matrix[i][j]) ? 'NaN' : 'Infinity';
        return `${errorType} detected in ${matrixName} at position [${i},${j}]`;
      }
    }
  }
  return null;
}
