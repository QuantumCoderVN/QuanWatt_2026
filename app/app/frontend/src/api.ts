// ============================================================
// api.ts
// API client for backend communication
// ============================================================

import type { PowerFlowCase, SolverSettings, ComparisonResult, Limits } from './types';

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

export async function healthCheck(): Promise<{ status: string }> {
  return fetchJSON(`${API_BASE}/health`);
}

export async function getDefaultCase(): Promise<{
  case: PowerFlowCase;
  limits: Limits;
  default_settings: SolverSettings;
}> {
  return fetchJSON(`${API_BASE}/case`);
}

export async function solvePowerFlow(
  powerCase: PowerFlowCase,
  settings?: SolverSettings
): Promise<ComparisonResult> {
  return fetchJSON(`${API_BASE}/solve`, {
    method: 'POST',
    body: JSON.stringify({
      case: powerCase,
      settings: settings || undefined,
    }),
  });
}

export async function getLimits(): Promise<Limits> {
  return fetchJSON(`${API_BASE}/limits`);
}
