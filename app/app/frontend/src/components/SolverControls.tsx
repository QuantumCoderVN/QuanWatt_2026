// ============================================================
// SolverControls.tsx
// Solver settings and run controls
// ============================================================

import { Play, Settings } from 'lucide-react';
import type { SolverSettings, Limits } from '../types';

interface Props {
  settings: SolverSettings;
  limits: Limits;
  onSettingsChange: (settings: SolverSettings) => void;
  onSolve: () => void;
  isRunning: boolean;
  disabled: boolean;
}

export default function SolverControls({
  settings,
  limits,
  onSettingsChange,
  onSolve,
  isRunning,
  disabled,
}: Props) {
  const handleChange = (field: keyof SolverSettings, value: number | boolean) => {
    onSettingsChange({ ...settings, [field]: value });
  };

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="w-5 h-5 text-gray-700" />
        <h2 className="text-xl font-semibold text-gray-900">Solver Settings</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Phase Qubits */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Phase Qubits
            <span className="text-gray-500 font-normal ml-1">
              ({limits.min_phase_qubits}–{limits.max_phase_qubits})
            </span>
          </label>
          <input
            type="number"
            value={settings.phase_qubits}
            onChange={(e) => handleChange('phase_qubits', parseInt(e.target.value))}
            className="input w-full"
            min={limits.min_phase_qubits}
            max={limits.max_phase_qubits}
            disabled={disabled}
          />
          <p className="text-xs text-gray-500 mt-1">
            Number of qubits for quantum phase estimation
          </p>
        </div>

        {/* Max Iterations */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Iterations
            <span className="text-gray-500 font-normal ml-1">
              ({limits.min_iterations}–{limits.max_iterations})
            </span>
          </label>
          <input
            type="number"
            value={settings.max_iter}
            onChange={(e) => handleChange('max_iter', parseInt(e.target.value))}
            className="input w-full"
            min={limits.min_iterations}
            max={limits.max_iterations}
            disabled={disabled}
          />
          <p className="text-xs text-gray-500 mt-1">
            Maximum FDLS iterations
          </p>
        </div>

        {/* Tolerance */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tolerance
          </label>
          <input
            type="number"
            value={settings.tolerance}
            onChange={(e) => handleChange('tolerance', parseFloat(e.target.value))}
            className="input w-full"
            step="1e-8"
            min="1e-10"
            disabled={disabled}
          />
          <p className="text-xs text-gray-500 mt-1">
            Convergence tolerance (p.u.)
          </p>
        </div>

        {/* Phase Target */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Phase Target
          </label>
          <input
            type="number"
            value={settings.phase_target}
            onChange={(e) => handleChange('phase_target', parseFloat(e.target.value))}
            className="input w-full"
            step="0.01"
            min="0.01"
            max="0.99"
            disabled={disabled}
          />
          <p className="text-xs text-gray-500 mt-1">
            HHL phase estimation target (0–1)
          </p>
        </div>

        {/* C Constant */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            C Constant
          </label>
          <input
            type="number"
            value={settings.C}
            onChange={(e) => handleChange('C', parseFloat(e.target.value))}
            className="input w-full"
            step="0.1"
            min="0.1"
            disabled={disabled}
          />
          <p className="text-xs text-gray-500 mt-1">
            Reciprocal rotation constant
          </p>
        </div>

        {/* Adaptive C */}
        <div className="flex items-center">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={settings.use_adaptive_c}
              onChange={(e) => handleChange('use_adaptive_c', e.target.checked)}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              disabled={disabled}
            />
            <span className="ml-2 text-sm font-medium text-gray-700">
              Use Adaptive C
            </span>
          </label>
          <p className="text-xs text-gray-500 mt-1 ml-6">
            Compute C from eigenvalues for custom networks
          </p>
        </div>
      </div>

      {/* Run Button */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <button
          onClick={onSolve}
          disabled={disabled || isRunning}
          className="btn btn-primary w-full md:w-auto flex items-center justify-center gap-2 text-lg py-3 px-8"
        >
          {isRunning ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              Running Solvers...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Run Power Flow Analysis
            </>
          )}
        </button>
        {isRunning && (
          <p className="text-sm text-gray-500 mt-2">
            This may take 30–60 seconds for HHL quantum simulation...
          </p>
        )}
      </div>
    </div>
  );
}
