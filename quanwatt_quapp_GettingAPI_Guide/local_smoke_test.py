"""Optional: run after pip install -r requirements.txt (no Quapp credentials)."""
import json
from pathlib import Path
import handler

payload = json.loads((Path(__file__).parent / 'sample_invocation.json').read_text())
circuit = handler.processing(payload)
assert circuit.__class__.__name__ == 'QuantumCircuit'
result = handler.post_processing(None)
assert result['schema_version'] == 'quanwatt-hhl-statevector-1'
assert result['request_id'] == payload['request_id']
assert result['execution_mode'] == 'quapp_statevector'
assert result['original_dimension'] == 2
assert result['padded_dimension'] == 2
assert len(result['branch_real']) == 2
assert 0 < result['postselection_rate'] <= 1
print('Qiskit return type:', type(circuit))
print('Sample output:', json.dumps(result, indent=2))
print('PASS: local handler contract check; cloud lifecycle still unverified.')
