// ============================================================
// ConvergenceChart.tsx
// Convergence history visualization (HHL only)
// ============================================================

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { SolverResult } from '../types';

interface Props {
  result: SolverResult;
}

export default function ConvergenceChart({ result }: Props) {
  const data = Array.from({ length: result.iterations }, (_, i) => ({
    iteration: i,
    mismatch: result.loss_history[i] ?? null,
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
            dataKey="mismatch"
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
