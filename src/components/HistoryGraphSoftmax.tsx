import React, { useMemo } from 'react';
import { HistorySoftMaxEntry } from '../App';

interface SoftmaxHistoryGraphProps {
  history: HistorySoftMaxEntry[];
  maxPoints?: number;
  vocabularyWords: string[];
  totalSteps: number;
}

const HistoryGraphSoftmax: React.FC<SoftmaxHistoryGraphProps> = ({
  history,
  maxPoints = 100,
  vocabularyWords,
  totalSteps,
}) => {
  // Get the last N points to display
  const displayHistory = useMemo(() => {
    return history.slice(-maxPoints);
  }, [history, maxPoints]);

  // SVG dimensions
  const width = window.innerWidth >= 1024 ? 600 : 300;
  const height = window.innerWidth >= 1024 ? 250 : 400;
  const padding = { top: 20, right: 20, bottom: 30, left: 45 };
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
      '#6366f1', // indigo
      '#d946ef', // purple
      '#f43f5e', // rose
      '#eab308', // yellow
      '#f97316', // orange
      '#0ea5e9', // sky
    ];

    return vocabularyWords.reduce((acc, word, idx) => {
      acc[word] = colors[idx % colors.length];
      return acc;
    }, {} as Record<string, string>);
  }, [vocabularyWords]);

  // Calculate min and max probabilities for auto-scaling
  const { minProb, maxProb } = useMemo(() => {
    if (displayHistory.length === 0) return { minProb: 0, maxProb: 1 };

    let min = 1;
    let max = 0;

    displayHistory.forEach((entry) => {
      entry.softmaxValues.forEach((sv) => {
        if (sv.probability < min) min = sv.probability;
        if (sv.probability > max) max = sv.probability;
      });
    });

    // Add some padding to make the graph more readable
    const range = max - min || 0.1;
    return {
      minProb: Math.max(0, min - range * 0.1),
      maxProb: Math.min(1, max + range * 0.1),
    };
  }, [displayHistory]);

  // Create path data for each token's probability line and track end positions
  const { pathsData, endPositions } = useMemo(() => {
    if (displayHistory.length === 0) return { pathsData: {}, endPositions: {} };

    const xScale = (i: number) =>
      (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
    const yScale = (prob: number) => {
      const normalized = (prob - minProb) / (maxProb - minProb || 1);
      return graphHeight - normalized * graphHeight;
    };

    const paths: Record<string, string> = {};
    const ends: Record<string, { x: number; y: number; prob: number }> = {};

    vocabularyWords.forEach((token) => {
      const points = displayHistory.map((entry, i) => {
        const tokenData = entry.softmaxValues.find((sv) => sv.token === token);
        const prob = tokenData?.probability ?? 0;
        const x = xScale(i);
        const y = yScale(prob);
        return { x, y, prob };
      });

      paths[token] = points
        .map((point, i) => `${i === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
        .join(' ');

      // Store the last point for label positioning
      if (points.length > 0) {
        const lastPoint = points[points.length - 1];
        ends[token] = lastPoint;
      }
    });

    return { pathsData: paths, endPositions: ends };
  }, [
    displayHistory,
    vocabularyWords,
    graphWidth,
    graphHeight,
    minProb,
    maxProb,
  ]);

  // Create Y-axis labels based on actual data range
  const yAxisLabels = useMemo(() => {
    const labels = [];
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
      const value = minProb + (maxProb - minProb) * (i / steps);
      const y = graphHeight - (i / steps) * graphHeight;
      labels.push({ value, y });
    }
    return labels;
  }, [graphHeight, minProb, maxProb]);

  if (history.length === 0) {
    return (
      <div className="mb-0.5 bg-white rounded p-0.5">
        <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
          Next Token Prediction Probabilities
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
        Next Token Prediction Probabilities
      </h3>
      <div className="p-1 sm:p-2">
        <div className="w-full">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto"
            preserveAspectRatio="xMidYMid meet"
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
                    className="text-sm sm:text-base fill-gray-600"
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

              {/* X-axis labels - only show the last step */}
              {displayHistory.length > 0 && (
                <text
                  x={graphWidth}
                  y={graphHeight + 15}
                  textAnchor="end"
                  className="text-sm sm:text-base fill-gray-600"
                >
                  {totalSteps}
                </text>
              )}

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

              {/* Floating labels at the end of each line */}
              {Object.entries(endPositions)
                .sort((a, b) => a[1].y - b[1].y) // Sort by y position to avoid overlaps
                .map(([token, position], idx, arr) => {
                  // Calculate adjusted y position to avoid overlaps
                  let adjustedY = position.y;
                  const minSpacing = 18;

                  // Check previous labels for overlap
                  for (let i = 0; i < idx; i++) {
                    const prevY = arr[i][1].y;
                    if (Math.abs(adjustedY - prevY) < minSpacing) {
                      adjustedY = prevY + minSpacing;
                    }
                  }

                  // Keep label within bounds
                  adjustedY = Math.max(
                    10,
                    Math.min(graphHeight - 10, adjustedY)
                  );

                  return (
                    <g key={`label-${token}`}>
                      {/* Line connecting label to data point */}
                      {adjustedY !== position.y && (
                        <line
                          x1={position.x}
                          y1={position.y}
                          x2={position.x + 5}
                          y2={adjustedY}
                          stroke={tokenColors[token]}
                          strokeWidth="1"
                          opacity="0.3"
                        />
                      )}
                      {/* Background rect for better readability */}
                      <rect
                        x={position.x + 5}
                        y={adjustedY - 8}
                        width={token.length * 7 + 4}
                        height={16}
                        fill="white"
                        opacity="0.9"
                        rx="2"
                      />
                      <text
                        x={position.x + 7}
                        y={adjustedY + 3}
                        className="text-xs font-medium"
                        fill={tokenColors[token]}
                      >
                        {token}
                      </text>
                    </g>
                  );
                })}

              {/* Axis labels */}
              <text
                x={graphWidth / 2}
                y={graphHeight + 25}
                textAnchor="middle"
                className="text-base sm:text-lg fill-gray-700 font-medium font-mono"
              >
                {'Time Steps'}
              </text>
              <text
                x={-graphHeight / 2}
                y={-45}
                textAnchor="middle"
                transform={`rotate(-90, ${-graphHeight / 2}, ${-45})`}
                className="text-base sm:text-lg fill-gray-700 font-medium"
              >
                Softmax Probability
              </text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
};

export default HistoryGraphSoftmax;
