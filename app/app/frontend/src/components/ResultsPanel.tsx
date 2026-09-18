// ============================================================
// ResultsPanel.tsx
// Display solver results and comparison
// ============================================================

import { AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';
import type { ComparisonResult } from '../types';
import { BUS_TYPE_NAMES } from '../types';
import ConvergenceChart from './ConvergenceChart';
import VoltageCharts from './VoltageCharts';

interface Props {
  results: ComparisonResult;
  isStale: boolean;
}

export default function ResultsPanel({ results, isStale }: Props) {
  const { hhl, classical, comparison } = results;

  const formatNumber = (value: number, decimals: number = 6) => {
    return value.toFixed(decimals);
  };

  const formatScientific = (value: number) => {
    return value.toExponential(2);
  };

  return (
    <div className="space-y-6">
      {/* Stale Warning */}
      {isStale && (
        <div className="card bg-yellow-50 border-yellow-200">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-900">Results are stale</h3>
              <p className="text-yellow-700 text-sm mt-1">
                Network has been modified. Run the solver again to update results.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Classical Results */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Classical Solver</h3>
            {classical.converged ? (
              <div className="flex items-center gap-1 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm font-medium">Converged</span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-yellow-600">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm font-medium">{classical.status}</span>
              </div>
            )}
          </div>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Iterations:</span>
              <span className="font-mono font-medium">{classical.iterations}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Final Mismatch:</span>
              <span className="font-mono font-medium">{formatScientific(classical.final_mismatch_pu)} p.u.</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Elapsed Time:</span>
              <span className="font-mono font-medium">{classical.elapsed_time_seconds.toFixed(4)} s</span>
            </div>
          </div>
        </div>

        {/* HHL Results */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">HHL Quantum Solver</h3>
            {hhl.converged ? (
              <div className="flex items-center gap-1 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm font-medium">Converged</span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-yellow-600">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm font-medium">{hhl.status}</span>
              </div>
            )}
          </div>

          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Iterations:</span>
              <span className="font-mono font-medium">{hhl.iterations}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Final Mismatch:</span>
              <span className="font-mono font-medium">{formatScientific(hhl.final_mismatch_pu)} p.u.</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Elapsed Time:</span>
              <span className="font-mono font-medium">{hhl.elapsed_time_seconds.toFixed(4)} s</span>
            </div>
          </div>

          {hhl.diagnostics && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs font-medium text-gray-700 mb-2">HHL Diagnostics:</p>
              <div className="space-y-1 text-xs text-gray-600">
                {hhl.diagnostics.bprime_scaling && (
                  <div>B' λ_bound: {hhl.diagnostics.bprime_scaling.lambda_bound?.toFixed(4)}</div>
                )}
                {hhl.diagnostics.bdouble_scaling && (
                  <div>B'' λ_bound: {hhl.diagnostics.bdouble_scaling.lambda_bound?.toFixed(4)}</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Metrics */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-gray-700" />
          <h3 className="text-lg font-semibold text-gray-900">HHL vs Classical Comparison</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-gray-600 mb-1">Max Voltage Magnitude Error</p>
            <p className="text-2xl font-bold text-gray-900">{formatScientific(comparison.max_vm_error)}</p>
            <p className="text-xs text-gray-500">p.u.</p>
          </div>
          <div>
            <p className="text-xs text-gray-600 mb-1">Max Voltage Angle Error</p>
            <p className="text-2xl font-bold text-gray-900">{formatScientific(comparison.max_va_error_deg)}</p>
            <p className="text-xs text-gray-500">degrees</p>
          </div>
          {comparison.first_theta_error !== null && (
            <div>
              <p className="text-xs text-gray-600 mb-1">First Δθ Error</p>
              <p className="text-2xl font-bold text-gray-900">{formatScientific(comparison.first_theta_error)}</p>
              <p className="text-xs text-gray-500">p.u.</p>
            </div>
          )}
          {comparison.first_v_error !== null && (
            <div>
              <p className="text-xs text-gray-600 mb-1">First ΔV Error</p>
              <p className="text-2xl font-bold text-gray-900">{formatScientific(comparison.first_v_error)}</p>
              <p className="text-xs text-gray-500">p.u.</p>
            </div>
          )}
        </div>

        {/* Validation Status */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-sm font-medium text-gray-700 mb-2">Reference Validation:</p>
          <div className="space-y-1 text-sm">
            {comparison.max_vm_error < 1e-7 ? (
              <div className="flex items-center gap-2 text-green-700">
                <CheckCircle className="w-4 h-4" />
                <span>Voltage magnitude within baseline tolerance (1e-7)</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-yellow-700">
                <AlertCircle className="w-4 h-4" />
                <span>Voltage magnitude exceeds baseline tolerance</span>
              </div>
            )}
            {comparison.max_va_error_deg < 1e-7 ? (
              <div className="flex items-center gap-2 text-green-700">
                <CheckCircle className="w-4 h-4" />
                <span>Voltage angle within baseline tolerance (1e-7)</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-yellow-700">
                <AlertCircle className="w-4 h-4" />
                <span>Voltage angle exceeds baseline tolerance</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Convergence Chart */}
      <ConvergenceChart classical={classical} hhl={hhl} />

      {/* Voltage Charts */}
      <VoltageCharts classical={classical} hhl={hhl} />

      {/* Bus Results Table */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Per-Bus Results</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Bus</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Vm (p.u.)</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Va (deg)</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">P (MW)</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Q (MVAr)</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">P Mismatch</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Q Mismatch</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {hhl.bus_results.map((bus, index) => (
                <tr key={bus.bus_id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-sm font-medium text-gray-900">{bus.bus_id}</td>
                  <td className="px-3 py-2 text-sm text-gray-600">{BUS_TYPE_NAMES[bus.bus_type]}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatNumber(bus.vm, 6)}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatNumber(bus.va_deg, 4)}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatNumber(bus.p_calc_mw, 2)}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatNumber(bus.q_calc_mvar, 2)}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatScientific(bus.p_mismatch_pu)}</td>
                  <td className="px-3 py-2 text-sm font-mono">{formatScientific(bus.q_mismatch_pu)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
