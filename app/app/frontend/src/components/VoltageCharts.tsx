// ============================================================
// VoltageCharts.tsx
// Voltage magnitude and angle comparison charts
// ============================================================

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { SolverResult } from '../types';

interface Props {
  classical: SolverResult;
  hhl: SolverResult;
}

export default function VoltageCharts({ classical, hhl }: Props) {
  // Prepare voltage magnitude data
  const vmData = classical.bus_results.map((bus, index) => ({
    bus: `Bus ${bus.bus_id}`,
    classical: bus.vm,
    hhl: hhl.bus_results[index].vm,
  }));

  // Prepare voltage angle data
  const vaData = classical.bus_results.map((bus, index) => ({
    bus: `Bus ${bus.bus_id}`,
    classical: bus.va_deg,
    hhl: hhl.bus_results[index].va_deg,
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Voltage Magnitude Chart */}
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
            <Bar dataKey="classical" fill="#059669" name="Classical" />
            <Bar dataKey="hhl" fill="#2563eb" name="HHL" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Voltage Angle Chart */}
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
            <Bar dataKey="classical" fill="#059669" name="Classical" />
            <Bar dataKey="hhl" fill="#2563eb" name="HHL" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
