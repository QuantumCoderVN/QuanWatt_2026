// ============================================================
// types.ts
// TypeScript type definitions
// ============================================================

export interface BusData {
  bus_id: number;
  bus_type: number;
  pd: number;
  qd: number;
  gs: number;
  bs: number;
  vm: number;
  va: number;
}

export interface GenData {
  gen_bus: number;
  pg: number;
  qg: number;
  vg: number;
}

export interface BranchData {
  f_bus: number;
  t_bus: number;
  r: number;
  x: number;
  b: number;
  tap: number;
  shift: number;
  status: number;
}

export interface PowerFlowCase {
  base_mva: number;
  bus: BusData[];
  gen: GenData[];
  branch: BranchData[];
}

export interface SolverSettings {
  phase_qubits: number;
  max_iter: number;
  tolerance: number;
  phase_target: number;
  C: number;
  use_adaptive_c: boolean;
}

export interface Limits {
  min_buses: number;
  max_buses: number;
  min_phase_qubits: number;
  max_phase_qubits: number;
  min_iterations: number;
  max_iterations: number;
}

export interface IterationResult {
  iteration: number;
  mismatch: number;
  mismatch_p: number[];
  mismatch_q: number[];
}

export interface BusResult {
  bus_id: number;
  bus_type: number;
  vm: number;
  va_deg: number;
  p_calc_mw: number;
  q_calc_mvar: number;
  p_spec_mw: number;
  q_spec_mvar: number;
  p_mismatch_pu: number;
  q_mismatch_pu: number;
}

export interface SolverResult {
  name: string;
  status: string;
  converged: boolean;
  iterations: number;
  final_mismatch_pu: number;
  elapsed_time_seconds: number;
  loss_history: number[];
  iteration_results: IterationResult[];
  bus_results: BusResult[];
  first_delta_theta: number[] | null;
  first_delta_v: number[] | null;
  settings: Record<string, any>;
  diagnostics?: Record<string, any>;
  error_message?: string;
}

export interface ComparisonResult {
  hhl: SolverResult;
  classical: SolverResult;
  comparison: {
    voltage_magnitude_errors: number[];
    voltage_angle_errors_deg: number[];
    max_vm_error: number;
    max_va_error_deg: number;
    first_theta_error: number | null;
    first_v_error: number | null;
  };
}

export const BUS_TYPE_NAMES: Record<number, string> = {
  1: 'PQ',
  2: 'PV',
  3: 'Slack',
};

export const DEFAULT_SETTINGS: SolverSettings = {
  phase_qubits: 8,
  max_iter: 8,
  tolerance: 1e-7,
  phase_target: 0.40,
  C: 1.0,
  use_adaptive_c: false,
};
