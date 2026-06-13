# Budget-Aware Neural Error Mitigation for Variational Quantum Algorithms

Code and data for the paper "Budget-Aware Neural Error Mitigation for
Variational Quantum Algorithms" (under review).

We train a small neural network that corrects noisy VQA expectation values
from a single shot-limited measurement. The model is conditioned on circuit
features, device-characterization features, and the shot-noise scale 1/sqrt(S),
so one network works across shot budgets. The main point of the paper is the
evaluation protocol: every method (ours, ZNE, CDR) gets the same total number
of shots per evaluation point, split across however many circuit executions
it needs. ZNE needs 3-5 executions, CDR needs 32 per target circuit, ours
needs one. At realistic per-iteration budgets this difference dominates.

## Setup

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m pytest tests/ -q
```

Needs Python >= 3.11. Everything runs on CPU.

## Running the experiments

Run from the repository root. Results are written as JSON under
`experiments/results/` (including per-instance predictions, which the paper's
tables and statistics are computed from).

```bash
# single cell of the scaling study
venv/bin/python experiments/scripts/scaling_cell.py --qubits 8 --regime miscal

# full campaign
bash experiments/scripts/run_campaign.sh "4 6 8 10 12"
bash experiments/scripts/run_campaign.sh "16"
bash experiments/scripts/run_20q.sh

# equal-budget sweep (Fig. 2)
venv/bin/python experiments/scripts/budget_sweep.py --regime miscal
venv/bin/python experiments/scripts/budget_sweep.py --regime systematic

# miscalibration phase boundary (Fig. 3)
venv/bin/python experiments/scripts/phase_boundary.py

# ablations and tail statistics
venv/bin/python experiments/scripts/feature_ablation.py
venv/bin/python experiments/scripts/model_size_ablation.py
venv/bin/python experiments/scripts/safeguard_analysis.py

# regenerate figures and appendix tables from the JSONs
venv/bin/python experiments/scripts/make_figures.py
venv/bin/python experiments/scripts/make_tables.py
```

The 20-qubit cells take roughly a day each on a 20-core machine; everything
else is minutes to a few hours.

## Code layout

```
src/quantum      circuits, noise models, angle-miscalibration transform
src/classical    ZNE and CDR baselines (mitiq), with shot accounting
src/models       mitigation network and the direct-prediction control
src/training     data generation and training loop
src/utils        exact and S-shot sampled expectation estimation
experiments/     entry points, configs, result JSONs
tests/           pytest suite
```

Two implementation details worth knowing about. `run_estimation_sampled`
(src/utils/qiskit_compat.py) estimates observables from actual measured
counts rather than noiseless-readout expectation values, which is what makes
the budget comparison meaningful. The miscalibration noise
(src/quantum/miscalibration.py) is applied to the executed circuit itself,
so folded ZNE circuits and CDR training circuits see the same calibration
state as the raw measurement; virtual RZ gates are exempt.

## Citation

```bibtex
@article{zhang2026budget,
  title  = {Budget-Aware Neural Error Mitigation for Variational Quantum Algorithms},
  author = {Zhang, Guilin and Zhao, Kai and Chu, Xu},
  year   = {2026},
  note   = {under review}
}
```

## License

MIT
