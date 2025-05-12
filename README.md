# Transformer Visualization

An interactive web application for visualizing transformer neural network attention mechanisms. This project aims to make transformer architectures more understandable by providing a visual representation of attention in transformer models.

## Features

- Visualize attention between tokens in transformer layers
- Interactive token selection to highlight attention patterns
- Support for visualizing multiple layers of a transformer
- Clean, modern UI built with React, TypeScript, and Tailwind CSS

## Technology Stack

- **React** - Frontend UI library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Create React App** - Build configuration and development setup

## Getting Started

### Prerequisites

- Node.js (14.x or later)
- npm or yarn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/transformer_visual.git
   cd transformer_visual
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

## Project Structure

```
src/
├── components/       # React components
│   ├── Token.tsx     # Individual token visualization
│   ├── AttentionLink.tsx  # Visualizes attention between tokens
│   └── Layer.tsx     # Layer visualization with tokens and attention
├── types/            # TypeScript type definitions
├── hooks/            # Custom React hooks
├── utils/            # Utility functions
└── App.tsx           # Main application component
```

## Future Enhancements

- Support for loading real transformer model data
- Visualization of embeddings using dimensionality reduction
- Animation of attention flows during inference
- Support for different transformer architectures (encoder-only, decoder-only, encoder-decoder)

## License

MIT

---

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).