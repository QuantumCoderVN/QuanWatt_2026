// ============================================================
// ConvergenceChart.tsx
// Convergence history visualization
// ============================================================

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { SolverResult } from '../types';

interface Props {
  classical: SolverResult;
  hhl: SolverResult;
}

export default function ConvergenceChart({ classical, hhl }: Props) {
  // Prepare data for chart
  const maxIterations = Math.max(classical.iterations, hhl.iterations);
  const data = Array.from({ length: maxIterations }, (_, i) => ({
    iteration: i,
    classical: classical.loss_history[i] || null,
    hhl: hhl.loss_history[i] || null,
  }));

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Convergence History</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="iteration"
            label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }}
          />
          <YAxis
            scale="log"
            domain={['auto', 'auto']}
            label={{ value: 'Mismatch (p.u.)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            formatter={(value: number) => value.toExponential(2)}
            labelFormatter={(label) => `Iteration ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="classical"
            stroke="#059669"
            name="Classical"
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="hhl"
            stroke="#2563eb"
            name="HHL"
            strokeWidth={2}
            dot={{ r: 4 }}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
