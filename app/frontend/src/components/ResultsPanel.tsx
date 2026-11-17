// ============================================================
// ResultsPanel.tsx
// Display HHL solver results
// ============================================================

import { AlertCircle, CheckCircle } from 'lucide-react';
import type { SolverResult } from '../types';
import { BUS_TYPE_NAMES } from '../types';
import ConvergenceChart from './ConvergenceChart';
import VoltageCharts from './VoltageCharts';

interface Props {
  results: SolverResult;
  isStale: boolean;
}

export default function ResultsPanel({ results, isStale }: Props) {
  const formatNumber = (value: number, decimals: number = 6) => {
    return value.toFixed(decimals);
  };

  const formatScientific = (value: number) => {
    return value.toExponential(2);
  };

  const executionMode =
    results.diagnostics?.bprime_scaling?.execution_mode ||
    results.diagnostics?.bdouble_scaling?.execution_mode;

  return (
    <div className="space-y-6">
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

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">HHL Quantum Solver</h3>
          {results.converged ? (
            <div className="flex items-center gap-1 text-green-600">
              <CheckCircle className="w-5 h-5" />
              <span className="text-sm font-medium">Converged</span>
            </div>
          ) : (
            <div className="flex items-center gap-1 text-yellow-600">
              <AlertCircle className="w-5 h-5" />
              <span className="text-sm font-medium">{results.status}</span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="flex justify-between sm:block">
            <span className="text-gray-600">Iterations</span>
            <p className="font-mono font-medium">{results.iterations}</p>
          </div>
          <div className="flex justify-between sm:block">
            <span className="text-gray-600">Final Mismatch</span>
            <p className="font-mono font-medium">{formatScientific(results.final_mismatch_pu)} p.u.</p>
          </div>
          <div className="flex justify-between sm:block">
            <span className="text-gray-600">Elapsed Time</span>
            <p className="font-mono font-medium">{results.elapsed_time_seconds.toFixed(4)} s</p>
          </div>
        </div>

        {results.status === 'timeout' && (
          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
            Solver stopped early due to time budget. Quapp HHL is slow (~1 min per linear
            system); increase iterations only if the proxy timeout allows it.
          </div>
        )}

        {results.diagnostics && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <p className="text-xs font-medium text-gray-700 mb-2">HHL / Quapp Diagnostics</p>
            <div className="space-y-1 text-xs text-gray-600">
              {executionMode && <div>First-solve execution mode: {executionMode}</div>}
              {results.diagnostics.quapp_solve_count != null && (
                <div>
                  Quapp solves: {results.diagnostics.quapp_solve_count}
                  {results.diagnostics.fallback_solve_count != null && (
                    <> · local fallback: {results.diagnostics.fallback_solve_count}</>
                  )}
                </div>
              )}
              {results.diagnostics.bprime_scaling && (
                <div>
                  B&apos; λ_bound: {results.diagnostics.bprime_scaling.lambda_bound?.toFixed(4)}
                  {results.diagnostics.bprime_scaling.residual != null && (
                    <> · residual: {formatScientific(results.diagnostics.bprime_scaling.residual)}</>
                  )}
                </div>
              )}
              {results.diagnostics.bdouble_scaling && (
                <div>
                  B&apos;&apos; λ_bound: {results.diagnostics.bdouble_scaling.lambda_bound?.toFixed(4)}
                  {results.diagnostics.bdouble_scaling.residual != null && (
                    <> · residual: {formatScientific(results.diagnostics.bdouble_scaling.residual)}</>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <ConvergenceChart result={results} />
      <VoltageCharts result={results} />

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
              {results.bus_results.map((bus) => (
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
