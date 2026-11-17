// ============================================================
// VoltageCharts.tsx
// Voltage magnitude and angle charts (HHL only)
// ============================================================

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { SolverResult } from '../types';

interface Props {
  result: SolverResult;
}

export default function VoltageCharts({ result }: Props) {
  const vmData = result.bus_results.map((bus) => ({
    bus: `Bus ${bus.bus_id}`,
    vm: bus.vm,
  }));

  const vaData = result.bus_results.map((bus) => ({
    bus: `Bus ${bus.bus_id}`,
    va: bus.va_deg,
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Voltage Magnitude</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={vmData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bus" />
            <YAxis
              domain={[0.9, 1.1]}
              label={{ value: 'Voltage (p.u.)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip formatter={(value: number) => value.toFixed(6)} />
            <Legend />
            <Bar dataKey="vm" fill="#2563eb" name="HHL" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Voltage Angle</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={vaData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bus" />
            <YAxis
              label={{ value: 'Angle (degrees)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip formatter={(value: number) => value.toFixed(4)} />
            <Legend />
            <Bar dataKey="va" fill="#2563eb" name="HHL" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
