# HHL multi-job workflow for Quapp

Bộ mã này tách script HHL ban đầu thành:

1. **Quapp function `hhl-main-circuit`**: một job đo mạch HHL để lấy `|x_i|²`.
2. **Quapp function `hhl-sign-circuit`**: ba job độc lập cho `(0,1)`, `(1,2)`, `(2,3)` để khôi phục dấu tương đối.
3. **Local controller**: invoke, poll, lưu JSON, hậu chọn, khôi phục dấu, tính hệ số `k`, so sánh nghiệm cổ điển và vẽ biểu đồ.

Mỗi thư mục function trên Quapp có đúng hai file: `handler.py` và `requirements.txt`.

## Bài toán cố định

```text
A = [[4.0,  0.4,  0.2,  0.0],
     [0.4,  5.0, -0.3,  0.1],
     [0.2, -0.3,  3.5,  0.5],
     [0.0,  0.1,  0.5,  4.5]]

b = [0.03, -0.02, 0.04, -0.01]
```

Nghiệm cổ điển tham chiếu xấp xỉ:

```text
[ 0.00732676, -0.00384886, 0.01116242, -0.00337696 ]
```

Dấu kỳ vọng của hướng nghiệm là `[+, -, +, -]`.

## Cấu trúc

```text
quapp/hhl-main/handler.py
quapp/hhl-main/requirements.txt
quapp/hhl-sign/handler.py
quapp/hhl-sign/requirements.txt
local/run_workflow.py
local/analyze_saved_results.py
local/offline_smoke_test.py
local/requirements.txt
local/requirements-dev.txt
local/.env.example
samples/*.json
```

## Deploy function 1: HHL main

- Tạo function tên `hhl-main-circuit`.
- Chọn/Open IDE với Qiskit runtime.
- Dán hai file trong `quapp/hhl-main/`.
- Chọn Qiskit SDK version tương thích với `qiskit>=1.2,<3`.
- Save version, Deploy và chờ trạng thái Ready.
- Test UI bằng Raw JSON `{}`.

Function trả về một `QuantumCircuit` 8 qubit và 8 classical bit. Measurement map qubit `q` sang classical bit `q`.

## Deploy function 2: HHL sign

- Tạo function tên `hhl-sign-circuit`.
- Chọn/Open IDE với Qiskit runtime.
- Dán hai file trong `quapp/hhl-sign/`.
- Save version, Deploy và chờ Ready.
- Test lần lượt ba Raw JSON trong `samples/`.

Mỗi sign job trả về mạch 9 qubit: 8 qubit HHL và `sign_extra` ở index 8.

## Tạo API token

Token phải được scope tới:

- cả hai function;
- device simulator/hardware định dùng;
- project hiện tại.

Lưu token vào `.env`, không ghi thẳng vào mã nguồn hoặc commit Git.

## Chạy local

```bash
cd local
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# sửa .env
python run_workflow.py
```

Controller tạo tuần tự bốn jobs, tải raw job detail, trích `jobResult`, lưu JSON và phân tích.

## Các file sinh ra ở local

```text
runs/<timestamp>/
  manifest.json
  jobs/
    hhl_invoke_response.json
    hhl_job_detail.json
    sign_0_1_invoke_response.json
    sign_0_1_job_detail.json
    sign_1_2_invoke_response.json
    sign_1_2_job_detail.json
    sign_2_3_invoke_response.json
    sign_2_3_job_detail.json
  results/
    hhl_result.json
    sign_0_1_result.json
    sign_1_2_result.json
    sign_2_3_result.json
  analysis/
    summary.json
  figures/
    solution_comparison.png
    probability_comparison.png
    sign_diagnostics.png
```

Quapp lưu **job record**, circuit view, histogram và JSON result. Các file trên là bản local do controller chủ động ghi xuống đĩa; Quapp không tự tạo cả cây thư mục này.

## Phân tích lại mà không chạy Quapp

```bash
python analyze_saved_results.py runs/<timestamp>
```

Lệnh này đọc bốn JSON trong `results/`, tính lại nghiệm và tạo lại biểu đồ. Vì vậy có thể thay đổi tiêu chí sign recovery hoặc visualization mà không tốn thêm quantum jobs.

## Smoke test bằng Aer trước khi deploy

```bash
pip install -r requirements-dev.txt
python offline_smoke_test.py
```

Aer chỉ chạy trong utility local. Hai Quapp handler hoàn toàn không import Aer, không transpile theo backend và không gọi `backend.run()`.

## Cách hậu xử lý

### Job HHL

Giữ shot thỏa:

```text
phase = 00000
ancilla = 1
```

Sau đó gom theo target 2 qubit để có `p_i`, chuẩn hóa điều kiện và lấy:

```text
|x_i| = sqrt(p_i)
```

### Job sign cho cặp `(i,j)`

Sau permutation, SWAP và Hadamard, local tính:

```text
2P(+) ≈ p_i + p_j + 2 x_i x_j
```

Do đó cross term:

```text
delta = 2P(+) - (p_i + p_j)
```

- `delta > 0`: cùng dấu;
- `delta < 0`: trái dấu;
- vùng `± tolerance`: chưa đủ chắc chắn do shot noise.

Đây là chỉnh sửa có cơ sở trực tiếp hơn so với so sánh `2P(+)` chỉ với `max(p_i,p_j)`.

### Khôi phục nghiệm không chuẩn hóa

Sau khi có hướng chuẩn hóa `x_direction`, tính:

```text
k = <A x_direction, b> / <A x_direction, A x_direction>
x_hhl = k x_direction
```

Sau đó so sánh `x_hhl` với `numpy.linalg.solve(A,b)` bằng absolute error, relative error, residual và fidelity hướng.

## Shots

- Kiểm tra wiring: 1,000–10,000 shots/job.
- Simulator đánh giá sơ bộ: 100,000 shots/job.
- Code nghiên cứu ban đầu: 1,000,000 shots/job.

Có 4 jobs, nên tổng số shots là `4 × shots_per_job`. Xác suất hậu chọn ancilla thường nhỏ; sign job cần đủ successful shots, không chỉ tổng shots lớn.
