# Transformer Architecture

## Introduction

This document describes the complete architecture of a transformer model, with particular focus on the attention mechanism and feed-forward networks (FFN) as implemented in this visualization project. The transformer architecture consists of stacked encoder and decoder layers, each containing attention mechanisms and feed-forward networks.

## Overall Transformer Structure

A standard transformer consists of:

1. **Encoder Stack**: Multiple identical encoder layers
2. **Decoder Stack**: Multiple identical decoder layers
3. **Input Embeddings**: Converts input tokens to vectors
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

## Multi-Head Attention Mechanism

### Structure

The attention mechanism in a transformer consists of several components working together:

1. **Multi-Head Attention**: Composed of multiple attention heads operating in parallel
2. **Query, Key, Value Projections**: Linear transformations of the input
3. **Scaled Dot-Product Attention**: The core attention operation
4. **Output Projection**: Combining outputs from all attention heads

### Mathematical Formulation

#### 1. Input Representation

Given an input sequence of tokens, each token is represented as an embedding vector of dimension `d_model`.

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

The Feed-Forward Network in a transformer is applied to each position in the sequence independently. It consists of two linear transformations with a non-linear activation function in between.

### Mathematical Formulation

For each position in the sequence:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

Or more generally:

```
FFN(x) = Activation(xW_1 + b_1)W_2 + b_2
```

Where:
- `W_1 ∈ ℝ^(d_model×d_ff)` and `b_1 ∈ ℝ^d_ff` are parameters of the first linear transformation
- `W_2 ∈ ℝ^(d_ff×d_model)` and `b_2 ∈ ℝ^d_model` are parameters of the second linear transformation
- `d_ff` is the inner dimension of the FFN (typically 4 times `d_model`)
- `Activation` is typically ReLU, but can also be GELU, Swish, etc.

The FFN can be viewed as a two-layer neural network with a ReLU (or other) activation.

## Complete Flow in a Transformer Layer

### Encoder Layer Flow

1. **Input**: Sequence of token embeddings `X`

2. **Self-Attention Sublayer**:
   - Apply multi-head self-attention: `Z = MultiHeadAttention(X, X, X)`
   - Add residual connection: `Z' = LayerNorm(X + Z)`

3. **Feed-Forward Sublayer**:
   - Apply position-wise FFN: `F = FFN(Z')`
   - Add residual connection: `F' = LayerNorm(Z' + F)`

4. **Output**: `F'` becomes the input to the next encoder layer or the final encoder output

### Decoder Layer Flow

1. **Input**: Sequence of token embeddings `Y` and encoder outputs `E`

2. **Masked Self-Attention Sublayer**:
   - Apply masked multi-head self-attention: `Z_1 = MaskedMultiHeadAttention(Y, Y, Y)`
   - Add residual connection: `Z_1' = LayerNorm(Y + Z_1)`

3. **Cross-Attention Sublayer**:
   - Apply multi-head cross-attention: `Z_2 = MultiHeadAttention(Z_1', E, E)`
   - Add residual connection: `Z_2' = LayerNorm(Z_1' + Z_2)`

4. **Feed-Forward Sublayer**:
   - Apply position-wise FFN: `F = FFN(Z_2')`
   - Add residual connection: `F' = LayerNorm(Z_2' + F)`

5. **Output**: `F'` becomes the input to the next decoder layer or the final decoder output

## Visualizing the Complete Transformer Flow

In our visualization:

1. **Token Embeddings**: Represented as vectors in the embedding space
2. **Attention Mechanism**:
   - Attention weights are shown as links between tokens
   - The brightness/thickness of links indicates attention strength
   - Multiple attention heads are visualized separately to show their different focus patterns
3. **Feed-Forward Network**:
   - Visualized as transformations applied to each token individually
   - Input and output projections are shown with their weights
   - Activation functions are represented visually

The complete flow visualization helps understand how:
- Information flows between tokens through the attention mechanism
- Each token's representation is transformed by the feed-forward network
- Residual connections help preserve information throughout the network
- Layer normalization stabilizes the learning process

This visualization demonstrates how transformers process relationships between tokens and how different components work together to enable the model's powerful capabilities in understanding and generating sequences.