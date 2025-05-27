import React, { useMemo } from 'react';
import { HistorySoftMaxEntry } from '../App';

interface SoftmaxHistoryGraphProps {
  history: HistorySoftMaxEntry[];
  maxPoints?: number;
  vocabularyWords: string[];
  totalSteps: number;
}

const SoftmaxHistoryGraph: React.FC<SoftmaxHistoryGraphProps> = ({
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
  const width = 800;
  const height = 200;
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

  // Create path data for each token's probability line
  const pathsData = useMemo(() => {
    if (displayHistory.length === 0) return {};

    const xScale = (i: number) =>
      (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
    const yScale = (prob: number) => {
      const normalized = (prob - minProb) / (maxProb - minProb || 1);
      return graphHeight - normalized * graphHeight;
    };

    const paths: Record<string, string> = {};

    vocabularyWords.forEach((token) => {
      const points = displayHistory.map((entry, i) => {
        const tokenData = entry.softmaxValues.find((sv) => sv.token === token);
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
          Next Token Prediction Probabilities Over Time
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
        Next Token Prediction Probabilities Over Time
      </h3>
      <div className="p-1 sm:p-2 flex gap-2">
        <div className="flex-auto min-w-0">
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

              {/* X-axis labels */}
              {displayHistory.length > 1 &&
                (() => {
                  const maxLabels = 5;
                  const step = Math.max(
                    1,
                    Math.floor(displayHistory.length / maxLabels)
                  );
                  const labels = [];

                  for (let i = 0; i < displayHistory.length; i += step) {
                    const x =
                      (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
                    const stepNumber =
                      totalSteps - displayHistory.length + i + 1;
                    labels.push(
                      <text
                        key={i}
                        x={x}
                        y={graphHeight + 15}
                        textAnchor="middle"
                        className="text-[9px] fill-gray-600"
                      >
                        {stepNumber}
                      </text>
                    );
                  }

                  // Always show the last step
                  if (displayHistory.length > 1) {
                    const lastX = graphWidth;
                    const lastStepNumber = totalSteps;
                    labels.push(
                      <text
                        key="last"
                        x={lastX}
                        y={graphHeight + 15}
                        textAnchor="middle"
                        className="text-[9px] fill-gray-600"
                      >
                        {lastStepNumber}
                      </text>
                    );
                  }

                  return labels;
                })()}

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
                className="text-[11px] fill-gray-700 font-medium font-mono"
              >
                {'Time Steps'}
              </text>
              <text
                x={-graphHeight / 2}
                y={-45}
                textAnchor="middle"
                transform={`rotate(-90, ${-graphHeight / 2}, ${-45})`}
                className="text-[11px] fill-gray-700 font-medium"
              >
                Softmax Probability
              </text>
            </g>
          </svg>
        </div>

        {/* Legend */}
        <div className="flex-shrink-0">
          <div className="bg-white border border-gray-200 rounded p-1 sm:p-2 text-[8px] sm:text-[10px] space-y-0.5 sm:space-y-1">
            {vocabularyWords.map((token, idx) => (
              <div key={token} className="flex items-center gap-0.5 sm:gap-1">
                <div
                  className="w-2 h-0.5 sm:w-3 sm:h-0.5"
                  style={{ backgroundColor: tokenColors[token] }}
                ></div>
                <span className="text-gray-700">{token}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SoftmaxHistoryGraph;
