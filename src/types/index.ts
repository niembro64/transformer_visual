export interface Attention {
  from: number;
  to: number;
  weight: number;
}

export interface TokenData {
  id: number;
  text: string;
  embedding?: number[];
}

export interface LayerData {
  id: number;
  name: string;
  tokens: TokenData[];
  attentions: Attention[];
}

export interface TransformerData {
  layers: LayerData[];
  sourceTokens: TokenData[];
  targetTokens: TokenData[];
}