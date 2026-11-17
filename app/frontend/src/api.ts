// ============================================================
// api.ts
// API client for backend / Quapp proxy
// ============================================================

import type {
  PowerFlowCase,
  SolverSettings,
  SolverResult,
  Limits,
  SolveProgressEvent,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new ApiError(error.detail || 'Request failed', response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError('Network error: ' + (error as Error).message);
  }
}

export async function getDefaultCase(): Promise<{
  case: PowerFlowCase;
  limits: Limits;
  default_settings: SolverSettings;
}> {
  return fetchJSON(`${API_BASE}/case`);
}

/** Stream progress events from /api/solve/stream (SSE over POST). */
export async function solvePowerFlowStream(
  powerCase: PowerFlowCase,
  settings: SolverSettings | undefined,
  onProgress: (event: SolveProgressEvent) => void
): Promise<SolverResult> {
  const response = await fetch(`${API_BASE}/solve/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      case: powerCase,
      settings: settings || undefined,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(error.detail || 'Request failed', response.status);
  }

  if (!response.body) {
    throw new ApiError('Streaming response not supported by this browser');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: SolverResult | null = null;
  let streamError: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';

    for (const chunk of chunks) {
      const line = chunk
        .split('\n')
        .map((l) => l.trim())
        .find((l) => l.startsWith('data:'));
      if (!line) continue;
      const raw = line.slice(5).trim();
      if (!raw) continue;

      let event: SolveProgressEvent;
      try {
        event = JSON.parse(raw);
      } catch {
        continue;
      }

      onProgress(event);

      if (event.type === 'result' && event.data) {
        finalResult = event.data;
      } else if (event.type === 'error') {
        streamError = event.message || 'Solver error';
      }
    }
  }

  if (streamError) {
    throw new ApiError(streamError);
  }
  if (!finalResult) {
    throw new ApiError('Solve stream ended without a result');
  }
  return finalResult;
}
