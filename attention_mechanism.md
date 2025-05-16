# Transformer Architecture

## Introduction

This document describes the complete architecture of a transformer model as specified in the original "Attention Is All You Need" paper (Vaswani et al., 2017), with particular focus on the attention mechanism and feed-forward networks (FFN). The transformer architecture consists of stacked encoder and decoder layers, each containing attention mechanisms and feed-forward networks.

## Overall Transformer Structure

A standard transformer consists of:

1. **Encoder Stack**: Multiple identical encoder layers
2. **Decoder Stack**: Multiple identical decoder layers
3. **Input Embeddings + Positional Encodings**: Converts input tokens to vectors with position information
4. **Output Linear Layer & Softmax**: Converts decoder outputs to probabilities

## Encoder Layer Structure

Each encoder layer contains two main sublayers:

1. **Multi-Head Self-Attention Mechanism**
2. **Position-wise Feed-Forward Network (FFN)**

Each sublayer has a residual connection and is followed by layer normalization.

## Decoder Layer Structure

Each decoder layer contains three main sublayers:

1. **Masked Multi-Head Self-Attention Mechanism**
2. **Multi-Head Cross-Attention Mechanism**
3. **Position-wise Feed-Forward Network (FFN)**

Each sublayer has a residual connection and is followed by layer normalization.

## Input Processing

### Token Embeddings

Input tokens are converted to embeddings of dimension `d_model`:

```
X = Embed(tokens)
```

### Positional Encodings

Since the transformer has no recurrence or convolution, positional information must be explicitly added:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence
- `i` is the dimension index
- Sine is used for even dimensions
- Cosine is used for odd dimensions

The positional encodings are added to the embeddings:

```
X' = X + PE
```

### Input Dropout

Dropout is applied after adding positional encodings:

```
X'' = Dropout(X')
```

## Multi-Head Attention Mechanism

### Structure

The attention mechanism in a transformer consists of several components working together:

1. **Multi-Head Attention**: Composed of multiple attention heads operating in parallel
2. **Query, Key, Value Projections**: Linear transformations of the input
3. **Scaled Dot-Product Attention**: The core attention operation
4. **Output Projection**: Combining outputs from all attention heads

### Mathematical Formulation

#### 1. Input Representation

Given an input sequence of tokens with positional encodings, each token is represented as an embedding vector of dimension `d_model`.

Input: `X ∈ ℝ^(n×d_model)` where:
- `n` is the sequence length
- `d_model` is the embedding dimension

#### 2. Query, Key, and Value Projections

For each attention head `i` (out of `h` heads):

- Query: `Q_i = XW_i^Q` where `W_i^Q ∈ ℝ^(d_model×d_k)`
- Key: `K_i = XW_i^K` where `W_i^K ∈ ℝ^(d_model×d_k)`
- Value: `V_i = XW_i^V` where `W_i^V ∈ ℝ^(d_model×d_v)`

Where:
- `d_k` is the dimension of queries and keys
- `d_v` is the dimension of values
- Typically `d_k = d_v = d_model/h`

#### 3. Scaled Dot-Product Attention

For each attention head:

1. Calculate attention scores: `S_i = Q_i K_i^T / √d_k`
2. Apply softmax to get attention weights: `A_i = softmax(S_i)`
3. Calculate weighted values: `Z_i = A_i V_i`

The formula for scaled dot-product attention is:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

The scaling factor `√d_k` prevents the dot products from growing too large in magnitude, which would push the softmax function into regions with extremely small gradients.

#### 4. Multi-Head Attention

The outputs from each attention head are concatenated and projected:

```
MultiHead(X) = Concat(Z_1, Z_2, ..., Z_h)W^O
```

Where `W^O ∈ ℝ^(h×d_v×d_model)` is the output projection matrix.

#### 5. Attention Dropout

Dropout is applied to the attention output before the residual connection:

```
Z' = Dropout(MultiHead(X))
```

### Self-Attention vs. Cross-Attention

- **Self-Attention**: When the queries, keys, and values all come from the same source (the input sequence)
- **Cross-Attention**: When queries come from one source (e.g., the decoder) and the keys and values come from another source (e.g., the encoder output)

### Masked Attention

In decoder self-attention, a mask is applied to prevent tokens from attending to future positions:

```
Masked_Attention(Q, K, V) = softmax(QK^T / √d_k + M)V
```

Where `M` is a mask with:
- `M[i,j] = 0` for `i ≥ j` (valid positions)
- `M[i,j] = -∞` for `i < j` (future positions)

## Feed-Forward Network (FFN)

### Structure

The Feed-Forward Network in a transformer is applied to each position in the sequence independently. It consists of two linear transformations with a ReLU activation function in between, followed by dropout.

### Mathematical Formulation

For each position in the sequence:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

More specifically with dropout:

```
F1 = xW_1 + b_1
F2 = ReLU(F1)
F3 = Dropout(F2)
FFN(x) = F3W_2 + b_2
```

Where:
- `W_1 ∈ ℝ^(d_model×d_ff)` and `b_1 ∈ ℝ^d_ff` are parameters of the first linear transformation
- `W_2 ∈ ℝ^(d_ff×d_model)` and `b_2 ∈ ℝ^d_model` are parameters of the second linear transformation
- `d_ff` is the inner dimension of the FFN (typically 4 times `d_model`)
- `ReLU(x) = max(0, x)` is the activation function

The FFN can be viewed as a two-layer neural network with a ReLU activation.

## Complete Flow in a Transformer Layer

### Encoder Layer Flow

1. **Input**: Sequence of token embeddings + positional encodings with dropout `X`

2. **Self-Attention Sublayer**:
   - Apply multi-head self-attention: `Z = MultiHeadAttention(X, X, X)`
   - Apply dropout: `Z_dropout = Dropout(Z)`
   - Add residual connection: `Z' = LayerNorm(X + Z_dropout)`

3. **Feed-Forward Sublayer**:
   - Apply position-wise FFN: `F = FFN(Z')` (includes internal dropout after ReLU)
   - Add residual connection: `F' = LayerNorm(Z' + F)`

4. **Output**: `F'` becomes the input to the next encoder layer or the final encoder output

### Decoder Layer Flow

1. **Input**: Sequence of token embeddings + positional encodings with dropout `Y` and encoder outputs `E`

2. **Masked Self-Attention Sublayer**:
   - Apply masked multi-head self-attention: `Z_1 = MaskedMultiHeadAttention(Y, Y, Y)`
   - Apply dropout: `Z_1_dropout = Dropout(Z_1)`
   - Add residual connection: `Z_1' = LayerNorm(Y + Z_1_dropout)`

3. **Cross-Attention Sublayer**:
   - Apply multi-head cross-attention: `Z_2 = MultiHeadAttention(Z_1', E, E)`
   - Apply dropout: `Z_2_dropout = Dropout(Z_2)`
   - Add residual connection: `Z_2' = LayerNorm(Z_1' + Z_2_dropout)`

4. **Feed-Forward Sublayer**:
   - Apply position-wise FFN: `F = FFN(Z_2')` (includes internal dropout after ReLU)
   - Add residual connection: `F' = LayerNorm(Z_2' + F)`

5. **Output**: `F'` becomes the input to the next decoder layer or the final decoder output

## Visualizing the Complete Transformer Flow

In our visualization:

1. **Token Embeddings + Positional Encodings**:
   - Token embeddings are represented as vectors in the embedding space
   - Sinusoidal positional encodings are added to incorporate position information
   - Dropout is applied after adding positional encodings (during training)

2. **Attention Mechanism**:
   - Attention weights are shown as links between tokens
   - The brightness/thickness of links indicates attention strength
   - Multiple attention heads are visualized separately to show their different focus patterns
   - Dropout is applied to attention outputs (during training)

3. **Feed-Forward Network**:
   - Visualized as transformations applied to each token individually
   - Input and output projections are shown with their weights
   - ReLU activation is applied after the first linear transformation
   - Dropout is applied after the ReLU activation (during training)

The complete flow visualization helps understand how:
- Information flows between tokens through the attention mechanism
- Each token's representation is transformed by the feed-forward network
- Residual connections help preserve information throughout the network
- Layer normalization stabilizes the learning process
- Dropout regularizes the model during training

This visualization demonstrates how transformers process relationships between tokens and how different components work together to enable the model's powerful capabilities in understanding and generating sequences.

## Implementation Changes Summary

To ensure compliance with the original "Attention Is All You Need" paper, we made the following changes:

1. **Sinusoidal Positional Encodings**:
   - Implemented `generatePositionalEncodings()` function that creates position-dependent encodings
   - Added `addPositionalEncodings()` to combine token embeddings with positional information
   - Added a visualization section showing raw embeddings, positional encodings, and their sum

2. **Dropout Layers**:
   - Added `applyDropout()` function with proper scaling during training
   - Implemented dropout after positional encodings are added to embeddings
   - Added dropout after the attention output before the residual connection
   - Added dropout after ReLU in the feed-forward network
   - Added a training mode toggle to enable/disable dropout visualization

3. **Feed-Forward Activation**:
   - Ensured ReLU is the default activation for feed-forward networks
   - Made activation function configurable to support alternative options
   - Updated visualization to clearly show the activation function in use

4. **UI Improvements**:
   - Added a "Training Mode" toggle to show the effect of dropout
   - Fixed multiple slider issue to ensure only one element can be edited at a time
   - Updated all component interfaces to maintain proper type safety
   - Enhanced matrix value visualization with sinusoidal color mapping
   - Made UI colors more vibrant with pure blue at +10 and pure red at -10

5. **Interactive Token Management**:
   - Added controls to add, remove, and edit tokens
   - Support for dynamic token count with automatic recalculation of all matrices
   - Ensured proper handling of tokens in all visualization components

6. **Dimension Adjustment**:
   - Added controls to increase/decrease embedding dimension
   - Automatic regeneration of all matrices when dimensions change
   - Real-time visualization updates when changing model dimensions

These changes ensure the implementation accurately follows the architecture described in the original transformer paper while providing an interactive, educational visualization that allows users to explore different transformer configurations.

## Interactive Controls Guide

### Token Management
- **Adding Tokens**: Click the "+" button in the token editor section to add a new empty token
- **Removing Tokens**: Hover over a token and click the "×" button that appears to remove that token
- **Editing Tokens**: Click on the token text field and type to modify the token's content

### Model Dimension Controls
- **Embedding Dimension**: Use the +/- buttons to increase or decrease the embedding dimension (d_model)
- The embedding dimension is kept even to ensure proper sinusoidal positional encodings
- When changing dimensions, all weights and matrices are regenerated appropriately

### Visualization Interaction
- Click on any editable matrix element (highlighted with a border on hover) to select it
- Use the slider that appears to adjust the selected value
- Click the "Oscillate" button to animate the selected value
- The color intensity reflects the value magnitude: blue for positive, red for negative
- Values are displayed in scientific notation for precision