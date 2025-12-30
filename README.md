# Generating brain-like networks with distance-dependent connection probability
Final project for the NAIL087 lecture at Charles University.

This project implements and analyzes two spatial network generation algorithms:
1. **Kaiser & Hilgetag (2004)** - Spatial growth model
2. **Vértes et al. (2012)** - Improved algorithm with fixed node positions

## Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

To run all analysis tasks and generate all required plots:

```bash
python analysis.py
```

This will execute:
- **Task 1.2**: Kaiser-Hilgetag plots for α=1, 5, 10 (β=0.5, N=100)
- **Task 1.3**: Kaiser-Hilgetag parameter scan with averaging
- **Task 1.4**: Small-world analysis
- **Task 2.2**: Vértes plots for α=5, 10, 20 (N=100)
- **Task 2.3**: Vértes parameter scan
- **Task 2.4**: Degree and weight distributions (α=15)
- **Task 3**: Mouse visual cortex comparison (if data available)

Results are saved to `./results_analysis/`.

## Running Individual Algorithms

You can also run the algorithm modules directly:

```bash
# Kaiser-Hilgetag original parameter scan
python kaiser_hilgetag.py

# Vértes original parameter scan
python vertes.py
```

These save results to `./results_kaiser/` and `./results_vertes/` respectively.

## Results

All experiment results are stored in this [Google Disk folder](https://drive.google.com/drive/folders/1s1UYKTHqOxMQqzLgWffSjGD5OiJ3TY3s?usp=drive_link)