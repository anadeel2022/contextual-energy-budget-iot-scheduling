Frozen V21 corrected-FIFO guarded contextual scheduler: independent long-run evaluation.

REQUIREMENTS
Windows PowerShell, Python 3.13; enough disk space for CSVs and figures.

INSTALL
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install matplotlib

UNIT TESTS
.\.venv\Scripts\python.exe .\test_packet_ledger.py
.\.venv\Scripts\python.exe .\test_deterministic_packet_service.py

SMOKE RUN (new output folder, two seeds, all six scenarios)
.\.venv\Scripts\python.exe .\run_final.py --seeds 2 --start 20001 --out outputs\smoke

LONG INDEPENDENT EVALUATION (200 untouched seeds; all six scenarios; 5 episodes each)
.\.venv\Scripts\python.exe .\run_final.py --seeds 200 --start 21001 --episodes 5 --out outputs\final_200

RERUN SAME COMMAND TO RESUME AFTER STOP/POWER LOSS
The protocol must be unchanged. Checkpoints save completed scenario/seed/policy groups.

OPTIONAL LARGER RUN (500 fresh seeds, separate output folder)
.\.venv\Scripts\python.exe .\run_final.py --seeds 500 --start 31001 --episodes 5 --out outputs\final_500

ANALYZE OR REGENERATE FIGURES FROM EXISTING COMPLETE RESULTS
.\.venv\Scripts\python.exe .\make_figures.py --out outputs\final_200

FILES
outputs\final_200\episodes.csv -- raw per episode results
outputs\final_200\summary.csv -- scenario and method averages; feasibility
outputs\final_200\paired_ci.csv -- paired 95% Student t confidence intervals
outputs\final_200\protocol.json -- immutable configuration and code SHA-256 checksums
outputs\final_200\figures\ -- publication-oriented PDF and PNG Figures 2-7
outputs\final_200\complete.json -- successful completion marker

IMPORTANT
This evaluation does NOT retrain or retune the controller. It uses frozen v19 fitted coefficients, V21 guard_0.0, and the original cap 0.2944864377500498 mJ/environment-step. The seed range must not overlap prior fitting, development or validation seeds. The six scenarios and 50-slot episode length match V21. A larger sample tightens estimates, not a mathematical guarantee of zero violations. Even if results differ from V21, report those differences and do not tune on the long-run evaluation seeds. Do not combine the smoke run with the long run as independent evidence if seeds overlap.

Note: same-model benchmarking as earlier V21, including corrected packet FIFO, abstract SINR physical-layer service, and common seed pairings. Run on Python 3.13 with NumPy, SciPy, PyTorch, pandas and matplotlib.
