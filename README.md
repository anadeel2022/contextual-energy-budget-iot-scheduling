# Contextual Energy-Budget-Aware Scheduling for Heterogeneous IoT Sensor Networks

Reproducibility materials for the corrected-FIFO V21 guarded contextual scheduler and the independent 200-seed evaluation. This archive includes the exact frozen source files used for the published evaluation records, unit tests, calibration artifacts, plotting scripts, and final CSV data. **No claims of formal budget guarantees are made.**

## Dataset and experiment

- 12 heterogeneous devices, one shared channel per environment, five parallel environments, 50 slots per episode and 1 ms per slot.
- Six scenarios: nominal, light traffic, heavy traffic, weaker channel, stronger channel, and heavy traffic plus weaker channel.
- Three policies: Guarded, Fixed9dB, Unguarded.
- 200 independent seeds, 21001 through 21200, five episodes each, giving 18,000 method-scenario-seed-episode records.
- Fixed energy allowance: `0.2944864377500498` mJ per environment-step. Frozen guard: `guard_0.0`.

## Layout

- `run_final.py`: final evaluator and resumable CSV checkpoint logic.
- `make_figures.py`: plotting module invoked by the evaluator.
- `make_figures_v22.py`: revised standalone plotting script for Figures 2–7.
- `code/`: exact frozen implementation, including `code/v19_fitted/model.json`. This retains supporting historical modules required by imports; their presence does not mean those policies were evaluated.
- `development_selection.json`: guard-development decision.
- `test_packet_ledger.py` and `test_deterministic_packet_service.py`: queue-accounting tests.
- `results/final_200/`: complete archived CSV results, protocol metadata, and PDF/PNG figures.
- `verify_archive.py`: quick completeness, uniqueness, and frozen-code hash audit.
- `SHA256SUMS.txt`: hashes for all packaged files.

## Environment and validation

Tested workflow targets Python 3.13. Install dependencies from `requirements.txt`.

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 .\test_packet_ledger.py
py -3.13 .\test_deterministic_packet_service.py
py -3.13 .\verify_archive.py
```

On macOS/Linux, use an installed Python 3.13 interpreter instead of `py -3.13`.

## Reproduce the simulation

```powershell
py -3.13 -u .\run_final.py --seeds 200 --start 21001 --episodes 5 --out outputs\replication_200
```

This regenerates the independent evaluation in a new output directory, without recalibrating or retraining the frozen controller. On Windows, close spreadsheet applications that are accessing the output CSV while checkpointing.

## Replot the archived data without rerunning the experiment

```powershell
py -3.13 .\make_figures_v22.py --out results\final_200
```

The archived published plots and data are stored in `results/final_200/figures/`. The replot command may overwrite figures in a local working copy; retain the distributed ZIP as an immutable reference.

## Caveats and provenance

The guard selects transmission modes using predicted energy rather than realized energy. The observed zero violations are empirical, not a mathematical safety guarantee. Statistical inference uses independent seed clusters, not episodes or parallel environments as independent samples. The simulator uses synthetic arrivals and an abstract SINR-dependent radio service. The data represent simulation measurements, not field measurements.

`results/final_200/protocol.json` contains immutable scenario, seed, model, and SHA-256 provenance information. `verify_archive.py` checks those exact frozen code hashes. The archive contains the final **research code and data**, not a fully integrated, newly compiled manuscript. An explicit software license has not yet been selected; choose one before publishing a public GitHub repository.
