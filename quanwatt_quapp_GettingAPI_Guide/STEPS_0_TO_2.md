# QuanWatt / Quapp deployment: README steps 0–2

## First, read this distinction

The existing `hhl_solver.py` is a **Qiskit** FDLS-backend integration, even though
its separate README describes **Perceval / Quandela**. Do **not** deploy the
Perceval runtime for this handler. Use **Qiskit** and a compatible simulator.
The private wizard link ending `functionId=155` cannot be inspected here.
If it was created with Perceval or Braket, make a NEW Qiskit function; the SDK
selection is immutable after creation.

**Crucial computational distinction:** This handler intentionally retains the
application's original `quanwatt-hhl-statevector-1` contract. It simulates the
HHL circuit on the Quapp Python worker via `Statevector.from_instruction`,
then submits a one-qubit placeholder circuit; `post_processing` returns cached
classical amplitudes and DOES NOT use quantum-device measurements. This does
not benchmark HHL on hardware. It additionally requires a shared process
between `processing` and `post_processing`, which is not guaranteed by public
Quapp docs, and serial invocation per worker. For genuine QPU execution,
change the readout design and the app client in a later phase.

## Step 0 — architecture and input

The app's UI/backend computes a linear system `A x = b` and calls:

```python
hhl_solve_with_diagnostics(A, b, phase_qubits=8,
                           phase_target=0.40, C=1.0)
```

The app's **existing** `hhl_solver.py` constructs the request (you do NOT
manually upload a circuit from the frontend):

```json
{
  "A": [[2.0, 0.0], [0.0, 1.0]],
  "b": [1.0, 1.0],
  "phase_qubits": 4,
  "phase_target": 0.40,
  "C": 0.25,
  "request_id": "manual-test-001"
}
```

The actual UI form fields and upstream mapping from buses/loads to `A,b` are
NOT in either attached file. This package matches the backend HHL input.
`A` must be real, symmetric, nonempty; `b` a real vector of the same size.
A non-power-of-two dimension is padded with an identity block. Inputs with
zero right-hand side are handled by the app locally, not sent to Quapp.
`phase_qubits` must be 1..8, `phase_target` strictly between 0 and 0.5,
`C > 0`. The parameters produce approximate HHL inverse phases, not an exact
linear solve. Use small matrices to begin with.

Backend REST workflow (later, after token is issued):

```text
UI / FDLS backend  -->  A, b, phase_qubits, phase_target, C
    -->  POST /api/v1/third-party/function/invoke  --> job ID
    -->  Quapp Qiskit processing(input) --> placeholder circuit
    -->  Quapp simulator runs placeholder --> post_processing(raw)
    -->  GET /v1/third-party/jobs/{job_id}/detail --> jobResult
    -->  app decodes branch and rescales it
```

## Step 1 — create and deploy on Quapp

1. Sign in at https://functions.quapp.cloud; select the app's Quapp project.
2. Open **Functions > Create Function** (or an existing Qiskit draft).
3. Name it `quanwatt-hhl-statevector` (lowercase and hyphens).
4. Choose **Open IDE** / **Blank Editor** (wording may vary).
5. Set `handler.py` exactly to this package's `handler.py`.
6. Set `requirements.txt` to the supplied file. Make sure the selected Qiskit
   runtime version is compatible with the pinned qiskit dependency; official
   Quapp Quickstart used **Qiskit 1.3.2** in October 2026.
7. Save/Sync the version, select **SDK / Environment: Qiskit** on Deploy,
   choose SDK version if prompted, and deploy that saved version.
8. Check **Ready** / successful deployment status and build logs.
9. Open the function's **Invocation** tab and choose a Qiskit-compatible
   simulator such as Quapp **aer_simulator** if available. Choose **json**
   input, paste `sample_invocation.json`, set **shots: 1**, and invoke.
10. Open Job Details. The expected `jobResult` has `schema_version`,
    `request_id`, `branch_real`, `branch_imag`, `postselection_rate`, etc.
    If you get `No cached HHL result`, Quapp's processing/post-processing
    processes are not sharing memory: this pattern is incompatible with that
    deployment and requires redesign.

**DO NOT choose Perceval/Quandela or Braket** for this Qiskit code. Quapp
requires the returned circuit type to match the selected SDK.

## Step 2 — create and record API Access Token

1. Be a Project Admin. Open **Project Settings > API Access > Create Token**.
2. Select `quanwatt-hhl-statevector` as an allowed function.
3. Select the **same Qiskit-compatible simulator** you used successfully
   in Step 1 (not an incompatible Quandela-only device).
4. Choose token name e.g. `quanwatt-backend-test`, expiration, description.
5. Review and create. Copy the **accessToken and refreshToken** once to a
   secure secrets manager; NEVER put them in source files, frontend JS,
   upload artifacts, screenshots, or chat messages.
6. Record `QUAPP_PROJECT_ID`, `QUAPP_FUNCTION_NAME`, `QUAPP_DEVICE_ID`,
   `QUAPP_TENANT_ID` (workspace ID; check your workspace or Quapp support).
   The README's `ws_seaquantathon2026` may be valid for that particular
   workspace; do not assume it universally.
7. For your existing app backend, set `QUAPP_API_TOKEN` equal to **accessToken**
   because `hhl_solver.py` uses that name. The README's `QUAPP_ACCESS_TOKEN`
   is for a DIFFERENT sample client. The app currently does not refresh its
   access token. Store **refreshToken** securely for a later client update.

You have **not created a new API endpoint**; Quapp's invocation and job-detail
endpoints already exist. The function name, device ID, workspace ID, project
ID, and access token allow your existing backend to call them.

## Compatibility issues found between README and app

- README's Perceval runtime / Quandela device is incompatible with Qiskit.
- README says `QUAPP_ACCESS_TOKEN`, but the app reads `QUAPP_API_TOKEN`.
- README GET path is `/api/v1/third-party/jobs/{job_id}/detail`; Quapp's
  current API Access Token page and your app use
  `/v1/third-party/jobs/{job_id}/detail`. Verify with Quapp if problems arise.
- The current app does not implement access-token refresh (its README sample
  client apparently does). A new token alone does not finish integration.
- The app has an authorized local fallback that may hide a remote failure in
  the UI. Inspect backend logs / execution_mode when testing Quapp.

## Source documentation

- https://docs.quapp.cloud/developer-documentation/quantum-function/create-function/
- https://docs.quapp.cloud/platform-capabilities/function-management/create-functions/
- https://docs.quapp.cloud/platform-capabilities/function-management/api-access-tokens/
- https://docs.quapp.cloud/developer-documentation/quantum-function/invoke-function/
