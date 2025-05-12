import React, { useState } from 'react';
import './App.css';
import Layer from './components/Layer';
import { TransformerData, LayerData, TokenData, Attention } from './types';

// Sample data for demonstration
const sampleTokens: TokenData[] = [
  { id: 1, text: "The" },
  { id: 2, text: "quick" },
  { id: 3, text: "brown" },
  { id: 4, text: "fox" },
  { id: 5, text: "jumps" },
  { id: 6, text: "over" },
  { id: 7, text: "the" },
  { id: 8, text: "lazy" },
  { id: 9, text: "dog" },
];

const sampleAttentions: Attention[] = [
  { from: 1, to: 7, weight: 0.8 },
  { from: 2, to: 4, weight: 0.6 },
  { from: 3, to: 4, weight: 0.9 },
  { from: 4, to: 9, weight: 0.7 },
  { from: 5, to: 6, weight: 0.5 },
  { from: 8, to: 9, weight: 0.95 },
];

const sampleLayers: LayerData[] = [
  {
    id: 1,
    name: "Layer 1 - Self Attention",
    tokens: sampleTokens,
    attentions: sampleAttentions,
  },
  {
    id: 2,
    name: "Layer 2 - Self Attention",
    tokens: sampleTokens,
    attentions: sampleAttentions.map(a => ({
      ...a,
      weight: a.weight * 0.8 // Different weights for layer 2
    })),
  }
];

const sampleData: TransformerData = {
  layers: sampleLayers,
  sourceTokens: sampleTokens,
  targetTokens: sampleTokens,
};

function App() {
  const [transformerData] = useState<TransformerData>(sampleData);
  
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold">Transformer Visualization</h1>
          <p className="text-blue-100">Interactive visualization of transformer model attention</p>
        </div>
      </header>
      
      <main className="container mx-auto p-4 relative">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4">Sample Transformer Model</h2>
          
          <div className="space-y-8">
            {transformerData.layers.map(layer => (
              <Layer key={layer.id} layer={layer} />
            ))}
          </div>
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