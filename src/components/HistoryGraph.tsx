import React, { useMemo } from 'react';
import { HistoryTrainingEntry } from '../App';

interface HistoryGraphProps {
  history: HistoryTrainingEntry[];
  maxPoints?: number;
}

const HistoryGraph: React.FC<HistoryGraphProps> = ({
  history,
  maxPoints = 100,
}) => {
  // Get the last N points to display
  const displayHistory = useMemo(() => {
    return history.slice(-maxPoints);
  }, [history, maxPoints]);

  // Calculate min and max loss for scaling
  const { minLoss, maxLoss } = useMemo(() => {
    if (displayHistory.length === 0) return { minLoss: 0, maxLoss: 1 };

    const losses = displayHistory.map((h) => h.loss);
    const min = Math.min(...losses);
    const max = Math.max(...losses);

    // Add some padding
    const range = max - min || 1;
    return {
      minLoss: Math.max(0, min - range * 0.1),
      maxLoss: max + range * 0.1,
    };
  }, [displayHistory]);

  // SVG dimensions
  const width = 680;
  const height = 150;
  const padding = { top: 20, right: 20, bottom: 30, left: 60 };
  const graphWidth = width - padding.left - padding.right;
  const graphHeight = height - padding.top - padding.bottom;

  // Create path data for the loss line
  const pathData = useMemo(() => {
    if (displayHistory.length === 0) return '';

    const xScale = (i: number) =>
      (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
    const yScale = (loss: number) => {
      const normalized = (loss - minLoss) / (maxLoss - minLoss || 1);
      return graphHeight - normalized * graphHeight;
    };

    return displayHistory
      .map((entry, i) => {
        const x = xScale(i);
        const y = yScale(entry.loss);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  }, [displayHistory, minLoss, maxLoss, graphWidth, graphHeight]);

  // Create Y-axis labels
  const yAxisLabels = useMemo(() => {
    const labels = [];
    const steps = 4;
    for (let i = 0; i <= steps; i++) {
      const value = minLoss + (maxLoss - minLoss) * (i / steps);
      const y = graphHeight - (i / steps) * graphHeight;
      labels.push({ value, y });
    }
    return labels;
  }, [minLoss, maxLoss, graphHeight]);

  // Get color for correct/incorrect predictions
  const getPointColor = (entry: HistoryTrainingEntry) => {
    return entry.targetToken === entry.predictedToken ? '#22c55e' : '#ef4444';
  };

  if (history.length === 0) {
    return (
      <div className="mb-0.5 bg-white rounded p-0.5">
        <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
          Cross-Entropy Loss Over Time
        </h3>
        <div className="p-2 text-center text-gray-500 text-xs italic">
          Start training to see loss history
        </div>
      </div>
    );
  }

  return (
    <div className="mb-0.5 bg-white rounded p-0.5">
      <h3 className="text-xs sm:text-sm font-semibold mb-0.5 border-b pb-0.5">
        Cross-Entropy Loss Over Time
      </h3>
      <div className="p-1 sm:p-2 flex gap-2">
        <div className="flex-[3] min-w-0">
          <svg
            viewBox={`0 0 ${width - 120} ${height}`}
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
                  {label.value.toFixed(2)}
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

            {/* Loss line */}
            <path
              d={pathData}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
              opacity="0.8"
            />

            {/* Points */}
            {displayHistory.map((entry, i) => {
              const x =
                (i / Math.max(1, displayHistory.length - 1)) * graphWidth;
              const y =
                graphHeight -
                ((entry.loss - minLoss) / (maxLoss - minLoss || 1)) *
                  graphHeight;

              return (
                <g key={i}>
                  <circle
                    cx={x}
                    cy={y}
                    r="3"
                    fill={getPointColor(entry)}
                    opacity="0.8"
                  />
                  {/* Hover tooltip */}
                  <g opacity="0">
                    <rect
                      x={x - 40}
                      y={y - 35}
                      width="80"
                      height="30"
                      fill="black"
                      fillOpacity="0.8"
                      rx="4"
                    />
                    <text
                      x={x}
                      y={y - 20}
                      textAnchor="middle"
                      className="text-[10px] fill-white"
                    >
                      {entry.predictedToken} → {entry.targetToken}
                    </text>
                    <text
                      x={x}
                      y={y - 8}
                      textAnchor="middle"
                      className="text-[10px] fill-white"
                    >
                      Loss: {entry.loss.toFixed(3)}
                    </text>
                    <animate
                      attributeName="opacity"
                      from="0"
                      to="1"
                      dur="0.2s"
                      begin="mouseover"
                      fill="freeze"
                    />
                    <animate
                      attributeName="opacity"
                      from="1"
                      to="0"
                      dur="0.2s"
                      begin="mouseout"
                      fill="freeze"
                    />
                  </g>
                </g>
              );
            })}

            {/* Axis labels */}
            <text
              x={graphWidth / 2}
              y={graphHeight + 25}
              textAnchor="middle"
              className="text-[11px] fill-gray-700 font-medium"
            >
              Time Steps (Last {displayHistory.length})
            </text>
            </g>
          </svg>
        </div>
        
        {/* Legend */}
        <div className="flex-1 flex items-start">
          <div className="bg-white border border-gray-200 rounded p-2 text-[10px] space-y-1">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="text-gray-700">Correct Token</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              <span className="text-gray-700">Wrong Token</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryGraph;
