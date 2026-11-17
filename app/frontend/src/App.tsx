// ============================================================
// App.tsx
// Main application component
// ============================================================

import { useState, useEffect } from 'react';
import { AlertCircle, Zap, Info } from 'lucide-react';
import type {
  PowerFlowCase,
  SolverSettings,
  SolverResult,
  Limits,
  SolveProgressEvent,
} from './types';
import { DEFAULT_SETTINGS } from './types';
import { getDefaultCase, solvePowerFlowStream } from './api';
import NetworkEditor from './components/NetworkEditor';
import SolverControls from './components/SolverControls';
import ResultsPanel from './components/ResultsPanel';
import SolveProgressPanel from './components/SolveProgressPanel';

type AppState = 'idle' | 'loading' | 'solving' | 'success' | 'error';

function App() {
  const [state, setState] = useState<AppState>('loading');
  const [case_, setCase] = useState<PowerFlowCase | null>(null);
  const [settings, setSettings] = useState<SolverSettings>(DEFAULT_SETTINGS);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [results, setResults] = useState<SolverResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStale, setIsStale] = useState(false);
  const [progressEvents, setProgressEvents] = useState<SolveProgressEvent[]>([]);
  const [currentProgress, setCurrentProgress] = useState<SolveProgressEvent | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    getDefaultCase()
      .then((data) => {
        setCase(data.case);
        setLimits(data.limits);
        setSettings({ ...DEFAULT_SETTINGS, ...data.default_settings });
        setState('idle');
      })
      .catch((err) => {
        setError(err.message);
        setState('error');
      });
  }, []);

  useEffect(() => {
    if (state !== 'solving') return;
    setElapsedSeconds(0);
    const id = window.setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);
    return () => window.clearInterval(id);
  }, [state]);

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
    setProgressEvents([]);
    setCurrentProgress({
      type: 'status',
      phase: 'starting',
      message: 'Connecting to solver proxy…',
    });

    try {
      const result = await solvePowerFlowStream(case_, settings, (event) => {
        if (event.type === 'status') {
          setCurrentProgress(event);
          setProgressEvents((prev) => [...prev, event]);
          if (typeof event.elapsed_s === 'number') {
            setElapsedSeconds(Math.max(0, Math.floor(event.elapsed_s)));
          }
        }
      });
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
                  Draw the network, then solve with Quapp HHL
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
            <div className="card bg-blue-50 border-blue-200">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-900">
                  <p>
                    <strong>HHL Quantum Solver:</strong> Edit the network below, then run
                    power flow. Linear systems are solved on Quapp HHL.
                    Supports {limits.min_buses}–{limits.max_buses} buses.
                  </p>
                </div>
              </div>
            </div>

            <NetworkEditor
              case={case_}
              limits={limits}
              onChange={handleCaseChange}
            />

            <SolverControls
              settings={settings}
              limits={limits}
              onSettingsChange={handleSettingsChange}
              onSolve={handleSolve}
              isRunning={state === 'solving'}
              disabled={state === 'loading'}
            />

            {state === 'solving' && (
              <SolveProgressPanel
                events={progressEvents}
                current={currentProgress}
                elapsedSeconds={elapsedSeconds}
                maxIter={settings.max_iter}
              />
            )}

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

            {results && state !== 'solving' && (
              <ResultsPanel
                results={results}
                isStale={isStale}
              />
            )}
          </div>
        )}
      </main>

      <footer className="mt-12 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            HHL Power Flow Dashboard • Quapp HHL linear solves
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
