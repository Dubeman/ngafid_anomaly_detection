# NGAFID Anomaly Detection

Machine learning pipeline for detecting anomalies in aviation flight data using the **National General Aviation Flight Information Database (NGAFID)**.

The repo focuses on time-series anomaly detection and forecasting-style modeling, with both classical ML baselines and deep/time-series approaches.

## What it does

Typical pipeline:

1. Load and preprocess flight time series
2. Feature engineering and normalization
3. Run anomaly detection models
4. Evaluate and visualize results

## Algorithms included

- Statistical outlier detection: `Z-score`, `IQR`
- Classical ML: `Isolation Forest`, `One-Class SVM`, `Local Outlier Factor`
- Deep learning: autoencoder-based approaches (see notebooks)
- Time-series methods: `ARIMA`, seasonal decomposition, change-point detection

## Repository structure

Top-level scripts and notebooks:

- `anomaly_detection.py`: main anomaly detection logic
- `rocket_forecasting.py`: forecasting workflow helpers
- `ROCKET.py`: ROCKET-related utilities/entry
- `utils/`: shared helpers
- `notebooks/` + standalone notebooks:
  - `NGAFID_MC_20210917.ipynb`
  - `ROCKET.ipynb`

## Setup

The repo includes an environment definition:

- `ngafid_env.yaml`

Use it to recreate the runtime environment (conda):

```bash
conda env create -f ngafid_env.yaml
conda activate <env-name>
```

Then run the notebooks with Jupyter:

```bash
jupyter lab
```

## Data

NGAFID is an external dataset. The notebooks/scripts assume you provide the relevant flight data files as configured in the notebook code.

## Reported performance

The original project README included headline metrics; actual results depend on dataset splits and experimental settings documented in the notebooks.

## Citation

If you use this work in research, please cite:

```bibtex
@misc{dubey2024ngafid,
  title={Machine Learning Approaches for Anomaly Detection in Aviation Flight Data},
  author={Dubey, Manas},
  year={2024},
  institution={Your Institution}
}
```

## License

MIT
