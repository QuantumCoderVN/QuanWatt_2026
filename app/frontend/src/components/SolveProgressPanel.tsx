// ============================================================
// SolveProgressPanel.tsx
// Live progress while waiting for Quapp HHL power-flow
// ============================================================

import { Loader2, CheckCircle2, Radio } from 'lucide-react';
import type { SolveProgressEvent } from '../types';

interface Props {
  events: SolveProgressEvent[];
  current: SolveProgressEvent | null;
  elapsedSeconds: number;
  maxIter: number;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s.toString().padStart(2, '0')}s` : `${s}s`;
}

export default function SolveProgressPanel({
  events,
  current,
  elapsedSeconds,
  maxIter,
}: Props) {
  const iteration = current?.iteration ?? 0;
  const solveIndex = current?.solve_index;
  const mismatch = current?.mismatch;
  const phase = current?.phase || 'starting';

  const progressPct =
    maxIter > 0 && iteration > 0
      ? Math.min(95, Math.round((iteration / maxIter) * 100))
      : phase === 'build_network'
        ? 5
        : phase === 'quapp_invoke' || phase === 'quapp_done'
          ? Math.min(90, 10 + (solveIndex || 1) * 8)
          : 8;

  const statusEvents = events.filter((e) => e.type === 'status').slice(-8);

  return (
    <div className="card border-primary-200 bg-white shadow-sm">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Loader2 className="w-7 h-7 text-primary-600 animate-spin" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Solving power flow</h3>
            <p className="text-sm text-gray-600 mt-0.5">
              {current?.message || 'Preparing request…'}
            </p>
          </div>
        </div>
        <div className="text-right shrink-0">
          <p className="text-2xl font-mono font-semibold text-primary-700">
            {formatElapsed(elapsedSeconds)}
          </p>
          <p className="text-xs text-gray-500">elapsed</p>
        </div>
      </div>

      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>
            {iteration > 0
              ? `Iteration ${iteration}${maxIter ? ` / ${maxIter}` : ''}`
              : 'Starting…'}
          </span>
          <span>{progressPct}%</span>
        </div>
        <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
          <div
            className="h-full rounded-full bg-primary-500 transition-all duration-500 ease-out"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4 text-sm">
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-xs text-gray-500">Phase</p>
          <p className="font-medium text-gray-900 truncate">{phase}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-xs text-gray-500">Linear solve #</p>
          <p className="font-mono font-medium text-gray-900">{solveIndex ?? '—'}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-xs text-gray-500">Mismatch</p>
          <p className="font-mono font-medium text-gray-900">
            {mismatch != null ? mismatch.toExponential(2) : '—'}
          </p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-xs text-gray-500">Last HHL mode</p>
          <p className="font-medium text-gray-900 truncate">
            {current?.execution_mode || '—'}
          </p>
        </div>
      </div>

      <div className="border-t border-gray-100 pt-3">
        <p className="text-xs font-medium text-gray-500 mb-2 flex items-center gap-1">
          <Radio className="w-3.5 h-3.5" />
          Live log
        </p>
        <ul className="space-y-1.5 max-h-40 overflow-y-auto text-sm">
          {statusEvents.length === 0 && (
            <li className="text-gray-400">Waiting for first progress event…</li>
          )}
          {statusEvents.map((ev, i) => (
            <li key={`${ev.elapsed_s}-${i}`} className="flex gap-2 text-gray-700">
              <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0 mt-0.5" />
              <span>
                <span className="font-mono text-xs text-gray-400 mr-2">
                  {ev.elapsed_s != null ? `${ev.elapsed_s}s` : ''}
                </span>
                {ev.message}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <p className="mt-4 text-xs text-gray-500">
        Each Quapp HHL call can take ~1 minute. Keep this tab open until results appear.
      </p>
    </div>
  );
}
