# EEG Cognitive Workload Classification

ML system for EEG-based workload classification.

## Results Summary
- **Best Accuracy**: 77.27% (Extremes, RF, Bg+W).

## Structure
- `ml_analysis/src/`: Core logic (Features, Data, Models).
- `docs/`: Documentation and diagrams.
- `data/`: Raw .mat files.

## Usage
```bash
python -m ml_analysis.main --config [trial|window] --task [binary|extremes|4level] --model [rf|svm] --features Bg,E,W,R,H
```

**Parameters:**
- `--config`: Segmentation `trial` or `window`.
- `--task`: Tasks `4level`, `binary` or `extremes`.
- `--model`: Algorithm `rf` or `svm`.
- `--eval`: Evaluation `ws` (intra-subject) or `group` (between-subjects).

## Features
- **B/Bg**: Band Power.
- **E**: Stats.
- **W**: Wavelets.
- **R**: Ratios.
- **H**: Hjorth.
- **N**: Entropy.
- **Z**: Zero-Crossing.

---
Data from: *"Estimating cognitive workload using a commercial in-ear EEG headset"*.
