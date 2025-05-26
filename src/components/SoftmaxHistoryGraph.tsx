import React, { useMemo } from 'react';
import { HistorySoftMaxEntry } from '../App';

interface SoftmaxHistoryGraphProps {
  history: HistorySoftMaxEntry[];
  maxPoints?: number;
  vocabularyWords: string[];
}

const SoftmaxHistoryGraph: React.FC<SoftmaxHistoryGraphProps> = ({
  history,
  maxPoints = 100,
  vocabularyWords
}) => {
  // Get the last N points to display
  const displayHistory = useMemo(() => {
    return history.slice(-maxPoints);
  }, [history, maxPoints]);

  // SVG dimensions
  const width = 800;
  const height = 200;
  const padding = { top: 20, right: 120, bottom: 30, left: 60 };
  const graphWidth = width - padding.left - padding.right;
  const graphHeight = height - padding.top - padding.bottom;

  // Create a color for each token
  const tokenColors = useMemo(() => {
    const colors = [
      '#ef4444', // red
      '#f59e0b', // amber
      '#84cc16', // lime
      '#22c55e', // green
      '#06b6d4', // cyan
      '#3b82f6', // blue
      '#8b5cf6', // violet
      '#ec4899', // pink
      '#f97316', // orange
      '#10b981', // emerald
    ];
    
    return vocabularyWords.reduce((acc, word, idx) => {
      acc[word] = colors[idx % colors.length];
      return acc;
    }, {} as Record<string, string>);
  }, [vocabularyWords]);

  // Create path data for each token's probability line
  const pathsData = useMemo(() => {
    if (displayHistory.length === 0) return {};
    
    const xScale = (i: number) => (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
    const yScale = (prob: number) => graphHeight - prob * graphHeight;

    const paths: Record<string, string> = {};
    
    vocabularyWords.forEach(token => {
      const points = displayHistory.map((entry, i) => {
        const tokenData = entry.softmaxValues.find(sv => sv.token === token);
        const prob = tokenData?.probability ?? 0;
        const x = xScale(i);
        const y = yScale(prob);
        return { x, y };
      });
      
      paths[token] = points
        .map((point, i) => `${i === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
        .join(' ');
    });
    
    return paths;
  }, [displayHistory, vocabularyWords, graphWidth, graphHeight]);

  // Create Y-axis labels
  const yAxisLabels = useMemo(() => {
    const labels = [];
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
      const value = i / steps;
      const y = graphHeight - (i / steps) * graphHeight;
      labels.push({ value, y });
    }
    return labels;
  }, [graphHeight]);

  if (history.length === 0) {
    return (
      <div className="mb-0.5 bg-white rounded p-0.5">
        <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
          Token Probability History
        </h3>
        <div className="p-2 text-center text-gray-500 text-xs italic">
          Start training to see probability history
        </div>
      </div>
    );
  }

  return (
    <div className="mb-0.5 bg-white rounded p-0.5">
      <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
        Token Probability History
      </h3>
      <div className="p-1 sm:p-2 overflow-x-auto">
        <svg 
          viewBox={`0 0 ${width} ${height}`} 
          className="w-full max-w-full h-auto"
          style={{ minWidth: '400px' }}
        >
          <g transform={`translate(${padding.left}, ${padding.top})`}>
            {/* Grid lines */}
            {yAxisLabels.map((label, i) => (
              <g key={i}>
                <line
                  x1={0}
                  y1={label.y}
                  x2={graphWidth}
                  y2={label.y}
                  stroke="#e5e7eb"
                  strokeWidth="1"
                />
                <text
                  x={-10}
                  y={label.y + 4}
                  textAnchor="end"
                  className="text-[10px] fill-gray-600"
                >
                  {(label.value * 100).toFixed(0)}%
                </text>
              </g>
            ))}

            {/* X-axis */}
            <line
              x1={0}
              y1={graphHeight}
              x2={graphWidth}
              y2={graphHeight}
              stroke="#9ca3af"
              strokeWidth="2"
            />

            {/* Y-axis */}
            <line
              x1={0}
              y1={0}
              x2={0}
              y2={graphHeight}
              stroke="#9ca3af"
              strokeWidth="2"
            />

            {/* Probability lines for each token */}
            {Object.entries(pathsData).map(([token, pathData]) => (
              <path
                key={token}
                d={pathData}
                fill="none"
                stroke={tokenColors[token]}
                strokeWidth="2"
                opacity="0.8"
              />
            ))}

            {/* Axis labels */}
            <text
              x={graphWidth / 2}
              y={graphHeight + 25}
              textAnchor="middle"
              className="text-[11px] fill-gray-700 font-medium"
            >
              Training Steps (Last {displayHistory.length})
            </text>
            <text
              x={-graphHeight / 2}
              y={-45}
              textAnchor="middle"
              transform={`rotate(-90, ${-graphHeight / 2}, ${-45})`}
              className="text-[11px] fill-gray-700 font-medium"
            >
              Probability
            </text>
          </g>

          {/* Legend */}
          <g transform={`translate(${width - 110}, 10)`}>
            <rect 
              x="0" 
              y="0" 
              width="100" 
              height={Math.min(vocabularyWords.length * 15 + 10, 170)} 
              fill="white" 
              stroke="#e5e7eb" 
              rx="4" 
            />
            {vocabularyWords.slice(0, 10).map((token, idx) => (
              <g key={token} transform={`translate(5, ${idx * 15 + 10})`}>
                <line
                  x1={0}
                  y1={0}
                  x2={15}
                  y2={0}
                  stroke={tokenColors[token]}
                  strokeWidth="2"
                />
                <text x={20} y={3} className="text-[10px] fill-gray-700">
                  {token}
                </text>
              </g>
            ))}
            {vocabularyWords.length > 10 && (
              <text x={5} y={165} className="text-[10px] fill-gray-600">
                ...{vocabularyWords.length - 10} more
              </text>
            )}
          </g>
        </svg>
      </div>
    </div>
  );
};

export default SoftmaxHistoryGraph;