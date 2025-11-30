# Transformer Load Analysis System

This system performs load forecasting, anomaly detection, and risk scoring for transformer-level data.

## Features

1. **Synthetic Dataset Generation**: Creates 60 days of transformer data at 15-minute intervals
2. **LSTM Load Forecasting**: Predicts next 6 hours of load using last 24 hours of data
3. **Anomaly Detection**: Uses Isolation Forest to detect abnormal load patterns
4. **Risk Scoring**: Computes transformer risk score based on load, temperature, and anomalies
5. **Visualizations**: Generates comprehensive plots showing predictions, anomalies, and risk scores

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python transformer_analysis.py
```

## Output Files

- `transformer_analysis_results.csv`: Final dataframe with predictions, anomalies, and risk scores
- `transformer_analysis_plots.png`: Main visualization with 3 subplots
- `lstm_test_predictions.png`: Detailed test set predictions

## Risk Score Formula

```
Risk Score = 0.6 * (Predicted_Load / Transformer_Capacity) 
           + 0.3 * Temperature_Factor 
           + 0.1 * Anomaly_Flag
```

Where:
- Transformer_Capacity = 100 kVA
- Temperature_Factor = (temperature - 25) / 20, scaled to 0-1

