import React, { useState } from 'react';
import './App.css';
import AttentionHead from './components/AttentionHead';
import FeedForward from './components/FeedForward';
import {
  generateSampleEmbeddings,
  generateSampleAttentionWeights,
  generateSampleMLPWeights
} from './utils/matrixOperations';

function App() {
  // Configuration for the demo
  const numTokens = 4; // Number of tokens in the sequence
  const embeddingDim = 8; // Dimension of token embeddings
  const attentionHeadDim = 6; // Dimension of attention head
  const mlpHiddenDim = 12; // Dimension of MLP hidden layer

  // Sample data generation
  const [embeddings] = useState(() => 
    generateSampleEmbeddings(numTokens, embeddingDim)
  );
  
  const [attentionWeights] = useState(() => 
    generateSampleAttentionWeights(embeddingDim, attentionHeadDim)
  );
  
  // Generate MLP weights that are compatible with attention head output dimensions
  const [mlpWeights] = useState(() => 
    generateSampleMLPWeights(embeddingDim, mlpHiddenDim, attentionHeadDim)
  );

  // State to hold the attention output context vectors
  const [attentionContext, setAttentionContext] = useState<number[][]>([]);

  // Token labels
  const tokenLabels = ["The", "cat", "sat", "down"];
  
  // Handler for receiving the computed context from the attention head
  const handleAttentionContextComputed = (context: number[][]) => {
    setAttentionContext(context);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold">Transformer Visualization</h1>
          <p className="text-blue-100">
            Interactive visualization of transformer attention mechanism and feed-forward network
          </p>
        </div>
      </header>
      
      <main className="container mx-auto p-4 pb-12">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">Transformer Components</h2>
          
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4 border-b pb-2">
              Part 1: Self-Attention Mechanism
            </h3>
            <p className="text-gray-700 mb-6">
              The self-attention mechanism allows each token to gather information from all other tokens
              in the sequence, weighting their relevance. This is how transformers capture long-range dependencies.
            </p>
            
            <AttentionHead
              embeddings={embeddings}
              weightQ={attentionWeights.weightQ}
              weightK={attentionWeights.weightK}
              weightV={attentionWeights.weightV}
              tokenLabels={tokenLabels}
              showSteps={true}
              onContextComputed={handleAttentionContextComputed}
            />
          </div>
          
          <div className="mt-12">
            <h3 className="text-xl font-semibold mb-4 border-b pb-2">
              Part 2: Feed-Forward Network (MLP)
            </h3>
            <p className="text-gray-700 mb-6">
              After the attention mechanism, each token representation passes through a 
              feed-forward neural network. This MLP is applied independently to each token position.
            </p>
            
            {attentionContext.length > 0 ? (
              <FeedForward
                inputs={attentionContext} // Using the output from the attention layer
                W1={mlpWeights.W1}
                b1={mlpWeights.b1}
                W2={mlpWeights.W2}
                b2={mlpWeights.b2}
                tokenLabels={tokenLabels}
                showSteps={true}
              />
            ) : (
              <div className="p-4 bg-gray-100 rounded">
                <p className="text-gray-600 italic">
                  The MLP visualization will appear here after the attention computation is complete.
                </p>
              </div>
            )}
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-4">About This Visualization</h2>
          <p className="text-gray-700 mb-4">
            This interactive visualization demonstrates the core components of a transformer architecture:
            the self-attention mechanism and the feed-forward network.
          </p>
          <p className="text-gray-700 mb-4">
            Transformers work by first using attention to gather context from the entire sequence,
            then processing that context through a feed-forward neural network at each position.
            This sequence of operations is typically repeated in multiple layers.
          </p>
          <p className="text-gray-700">
            The colors in the matrices represent values: blue for positive, red for negative, and white for near-zero.
            Brighter colors indicate stronger values.
          </p>
        </div>
      </main>
      
      <footer className="bg-gray-800 text-white p-4">
        <div className="container mx-auto text-center">
          <p>Transformer Visualization Demo</p>
        </div>
      </footer>
    </div>
  );
}

export default App;