// ============================================================
// App.tsx
// Main application component
// ============================================================

import { useState, useEffect } from 'react';
import { AlertCircle, Zap, Info } from 'lucide-react';
import type { PowerFlowCase, SolverSettings, ComparisonResult, Limits } from './types';
import { DEFAULT_SETTINGS } from './types';
import { getDefaultCase, solvePowerFlow } from './api';
import NetworkEditor from './components/NetworkEditor';
import SolverControls from './components/SolverControls';
import ResultsPanel from './components/ResultsPanel';

type AppState = 'idle' | 'loading' | 'solving' | 'success' | 'error';

function App() {
  const [state, setState] = useState<AppState>('loading');
  const [case_, setCase] = useState<PowerFlowCase | null>(null);
  const [settings, setSettings] = useState<SolverSettings>(DEFAULT_SETTINGS);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [results, setResults] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStale, setIsStale] = useState(false);

  // Load default case on mount
  useEffect(() => {
    getDefaultCase()
      .then((data) => {
        setCase(data.case);
        setLimits(data.limits);
        setSettings(data.default_settings);
        setState('idle');
      })
      .catch((err) => {
        setError(err.message);
        setState('error');
      });
  }, []);

  const handleCaseChange = (newCase: PowerFlowCase) => {
    setCase(newCase);
    setIsStale(true);
  };

  const handleSettingsChange = (newSettings: SolverSettings) => {
    setSettings(newSettings);
  };

  const handleSolve = async () => {
    if (!case_) return;

    setState('solving');
    setError(null);
    setIsStale(false);

    try {
      const result = await solvePowerFlow(case_, settings);
      setResults(result);
      setState('success');
    } catch (err: any) {
      setError(err.message || 'Failed to solve');
      setState('error');
    }
  };

  const busCount = case_?.bus.length || 0;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Zap className="w-8 h-8 text-primary-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  HHL Power Flow Dashboard
                </h1>
                <p className="text-sm text-gray-500">
                  Quantum linear solver for power flow analysis
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full font-medium">
                {busCount} {busCount === 1 ? 'Bus' : 'Buses'}
              </div>
              {isStale && results && (
                <div className="flex items-center gap-1 px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full">
                  <AlertCircle className="w-4 h-4" />
                  <span>Results stale</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {state === 'loading' && (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading application...</p>
            </div>
          </div>
        )}

        {state === 'error' && !case_ && (
          <div className="card bg-red-50 border-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-900">Error Loading Application</h3>
                <p className="text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {case_ && limits && (
          <div className="space-y-6">
            {/* Info Banner */}
            <div className="card bg-blue-50 border-blue-200">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-900">
                  <p>
                    <strong>HHL Quantum Solver:</strong> Demonstrates quantum linear system solving
                    for power flow analysis. Compare HHL results with classical solver.
                    Supports {limits.min_buses}–{limits.max_buses} buses.
                  </p>
                </div>
              </div>
            </div>

            {/* Network Editor */}
            <NetworkEditor
              case={case_}
              limits={limits}
              onChange={handleCaseChange}
            />

            {/* Solver Controls */}
            <SolverControls
              settings={settings}
              limits={limits}
              onSettingsChange={handleSettingsChange}
              onSolve={handleSolve}
              isRunning={state === 'solving'}
              disabled={state === 'loading'}
            />

            {/* Error Display */}
            {state === 'error' && error && (
              <div className="card bg-red-50 border-red-200">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-red-900">Solver Error</h3>
                    <p className="text-red-700 mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Results Panel */}
            {results && (
              <ResultsPanel
                results={results}
                isStale={isStale}
              />
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            HHL Power Flow Dashboard • Quantum Computing for Power Systems •{' '}
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:text-primary-700"
            >
              View Source
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
