"""
Transformer Load Forecasting, Anomaly Detection, and Risk Scoring System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, fallback to scikit-learn if it fails
USE_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    # Test if TensorFlow actually works (catches DLL errors)
    _ = tf.constant(1)
    USE_TENSORFLOW = True
    tf.random.set_seed(42)
    print("TensorFlow loaded successfully. Using LSTM model.")
except (ImportError, Exception) as e:
    print(f"TensorFlow not available or failed to load: {type(e).__name__}")
    print("Using MLPRegressor (Neural Network) instead.")
    print("Note: For LSTM model, install TensorFlow 2.15+ and Microsoft Visual C++ Redistributable.")

# Set random seeds for reproducibility
np.random.seed(42)

# Configuration
TRANSFORMER_CAPACITY = 100  # kVA
DAYS = 60
INTERVAL_MINUTES = 15
SAMPLES_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 96 samples per day
TOTAL_SAMPLES = DAYS * SAMPLES_PER_DAY

print("=" * 80)
print("Transformer Load Analysis System")
print("=" * 80)

# ============================================================================
# 1. GENERATE SYNTHETIC DATASET
# ============================================================================
print("\n[1/6] Generating synthetic dataset...")

def generate_synthetic_data():
    """Generate 60 days of transformer-level data at 15-minute intervals"""
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(minutes=i*INTERVAL_MINUTES) 
                  for i in range(TOTAL_SAMPLES)]
    
    # Generate temperature (seasonal pattern with daily variation)
    day_of_year = np.array([(t - start_date).days for t in timestamps])
    hour_of_day = np.array([t.hour + t.minute/60 for t in timestamps])
    
    # Base temperature: seasonal (20-35°C) + daily cycle
    base_temp = 25 + 7 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
    daily_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Daily
    temperature = base_temp + daily_variation + np.random.normal(0, 2, TOTAL_SAMPLES)
    temperature = np.clip(temperature, 15, 40)
    
    # Generate humidity (inverse relationship with temperature)
    humidity = 60 - 0.8 * (temperature - 25) + np.random.normal(0, 5, TOTAL_SAMPLES)
    humidity = np.clip(humidity, 30, 90)
    
    # Generate solar generation (0 at night, peak at noon)
    solar_generation = np.zeros(TOTAL_SAMPLES)
    for i, h in enumerate(hour_of_day):
        if 6 <= h <= 18:  # Daylight hours
            # Peak at noon (12:00)
            solar_generation[i] = 50 * np.sin(np.pi * (h - 6) / 12)
            solar_generation[i] = max(0, solar_generation[i])
    solar_generation += np.random.normal(0, 2, TOTAL_SAMPLES)
    solar_generation = np.clip(solar_generation, 0, 50)
    
    # Generate festival flags (3-4 random days)
    festival_days = np.random.choice(DAYS, size=np.random.randint(3, 5), replace=False)
    festival_flag = np.zeros(TOTAL_SAMPLES)
    for day in festival_days:
        start_idx = day * SAMPLES_PER_DAY
        end_idx = (day + 1) * SAMPLES_PER_DAY
        festival_flag[start_idx:end_idx] = 1
    
    # Generate base load (higher during high temperature)
    base_load = 40 + 15 * (temperature - 25) / 15  # Higher load when hot
    base_load = np.clip(base_load, 20, 80)
    
    # Add random spikes
    random_spikes = np.random.exponential(10, TOTAL_SAMPLES) * np.random.binomial(1, 0.1, TOTAL_SAMPLES)
    
    # Add festival spikes (higher load on festival days)
    festival_spikes = festival_flag * np.random.exponential(15, TOTAL_SAMPLES)
    
    # Calculate net load (load - solar generation)
    load_kW = base_load + random_spikes + festival_spikes
    net_load = load_kW - solar_generation * 0.3  # Solar reduces net load slightly
    net_load = np.clip(net_load, 10, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load_kW': net_load,
        'temperature': temperature,
        'humidity': humidity,
        'solar_generation_kW': solar_generation,
        'festival_flag': festival_flag
    })
    
    return df

df = generate_synthetic_data()
print(f"✓ Generated {len(df)} samples ({DAYS} days at {INTERVAL_MINUTES}-minute intervals)")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Load range: {df['load_kW'].min():.2f} - {df['load_kW'].max():.2f} kW")
print(f"  Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")

# ============================================================================
# 2. BUILD LSTM LOAD FORECASTING MODEL
# ============================================================================
print("\n[2/6] Building LSTM load forecasting model...")

def create_sequences(data, seq_length, forecast_horizon):
    """Create sequences for LSTM: last seq_length hours -> next forecast_horizon hours"""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

# Prepare data for LSTM
# Use last 24 hours (96 samples at 15-min intervals) to predict next 6 hours (24 samples)
SEQ_LENGTH = 96  # 24 hours
FORECAST_HORIZON = 24  # 6 hours

# Create separate scaler for load_kW (for inverse transform)
load_scaler = MinMaxScaler()
load_scaled = load_scaler.fit_transform(df[['load_kW']])

# Create sequences
X, y = create_sequences(load_scaled.flatten(), SEQ_LENGTH, FORECAST_HORIZON)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Further split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Test samples: {len(X_test)}")

if USE_TENSORFLOW:
    # Build LSTM model using TensorFlow
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16),
        Dense(FORECAST_HORIZON)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    print(f"✓ Model trained (best val loss: {min(history.history['val_loss']):.4f})")
    
    # Make predictions
    y_pred = model.predict(X_test_lstm, verbose=0)
    
    # Inverse transform predictions (first hour of forecast)
    predicted_load = load_scaler.inverse_transform(y_pred[:, 0].reshape(-1, 1)).flatten()
    
    # Get actual load for comparison (first hour of forecast)
    actual_load = []
    for i in range(len(X_test)):
        idx = train_size + i + SEQ_LENGTH
        if idx < len(df):
            actual_load.append(df.iloc[idx]['load_kW'])
    
    actual_load = np.array(actual_load)
    
    # For full dataset predictions
    print("  Generating predictions for full dataset...")
    all_predictions_scaled = []
    for i in range(len(load_scaled) - SEQ_LENGTH - FORECAST_HORIZON + 1):
        seq = load_scaled[i:i+SEQ_LENGTH].reshape(1, SEQ_LENGTH, 1)
        pred = model.predict(seq, verbose=0)
        all_predictions_scaled.append(pred[0, 0])  # First hour prediction
    
    # Inverse transform predictions
    all_predictions = load_scaler.inverse_transform(
        np.array(all_predictions_scaled).reshape(-1, 1)
    ).flatten()
    
else:
    # Use MLPRegressor (scikit-learn) as alternative
    print("  Using MLPRegressor (Neural Network) for forecasting...")
    
    # Train model for first hour prediction only
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        verbose=False
    )
    
    # Train on first hour prediction
    y_train_first = y_train[:, 0]
    y_val_first = y_val[:, 0]
    
    model.fit(X_train, y_train_first)
    
    # Validation score
    val_score = model.score(X_val, y_val_first)
    print(f"✓ Model trained (validation R²: {val_score:.4f})")
    
    # Make predictions
    y_pred_first = model.predict(X_test)
    predicted_load = load_scaler.inverse_transform(y_pred_first.reshape(-1, 1)).flatten()
    
    # Get actual load for comparison
    actual_load = []
    for i in range(len(X_test)):
        idx = train_size + i + SEQ_LENGTH
        if idx < len(df):
            actual_load.append(df.iloc[idx]['load_kW'])
    
    actual_load = np.array(actual_load)
    
    # For full dataset predictions
    print("  Generating predictions for full dataset...")
    all_predictions_scaled = []
    for i in range(len(load_scaled) - SEQ_LENGTH - FORECAST_HORIZON + 1):
        seq = load_scaled[i:i+SEQ_LENGTH].reshape(1, -1)
        pred = model.predict(seq)
        all_predictions_scaled.append(pred[0])
    
    # Inverse transform predictions
    all_predictions = load_scaler.inverse_transform(
        np.array(all_predictions_scaled).reshape(-1, 1)
    ).flatten()

# Pad with NaN for initial sequences
predicted_load_full = np.full(len(df), np.nan)
predicted_load_full[SEQ_LENGTH:SEQ_LENGTH+len(all_predictions)] = all_predictions

print("✓ Load forecasting complete")

# ============================================================================
# 3. BUILD ANOMALY DETECTION MODEL
# ============================================================================
print("\n[3/6] Building anomaly detection model...")

# Prepare features for anomaly detection
anomaly_features = ['load_kW', 'temperature', 'humidity', 'solar_generation_kW']
X_anomaly = df[anomaly_features].values

# Scale features
scaler_anomaly = MinMaxScaler()
X_anomaly_scaled = scaler_anomaly.fit_transform(X_anomaly)

# Use Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = isolation_forest.fit_predict(X_anomaly_scaled)

# Convert to binary (1 = anomaly, 0 = normal)
anomaly_labels = (anomaly_labels == -1).astype(int)

print(f"✓ Anomaly detection complete")
print(f"  Detected {anomaly_labels.sum()} anomalies ({100*anomaly_labels.sum()/len(anomaly_labels):.2f}%)")

# ============================================================================
# 4. COMPUTE TRANSFORMER RISK SCORE
# ============================================================================
print("\n[4/6] Computing Transformer Risk Score...")

def compute_risk_score(predicted_load, temperature, anomaly_flag, capacity=100):
    """Compute risk score based on load, temperature, and anomalies"""
    # Load factor (0-1)
    load_factor = np.clip(predicted_load / capacity, 0, 1)
    
    # Temperature factor (0-1)
    # (temperature - 25) / 20 scaled to 0-1
    temp_factor = (temperature - 25) / 20
    temp_factor = np.clip(temp_factor, 0, 1)
    
    # Anomaly flag (0 or 1)
    anomaly_factor = anomaly_flag
    
    # Risk score
    risk_score = 0.6 * load_factor + 0.3 * temp_factor + 0.1 * anomaly_factor
    
    return np.clip(risk_score, 0, 1)

# Handle NaN predictions (use actual load for initial periods)
predicted_for_risk = predicted_load_full.copy()
nan_mask = np.isnan(predicted_for_risk)
predicted_for_risk[nan_mask] = df.loc[nan_mask, 'load_kW'].values

risk_score = compute_risk_score(
    predicted_for_risk,
    df['temperature'].values,
    anomaly_labels
)

print("✓ Risk score computed")
print(f"  Risk score range: {risk_score.min():.3f} - {risk_score.max():.3f}")
print(f"  High risk periods (>0.7): {(risk_score > 0.7).sum()} ({100*(risk_score > 0.7).sum()/len(risk_score):.2f}%)")

# ============================================================================
# 5. CREATE FINAL DATAFRAME
# ============================================================================
print("\n[5/6] Creating final output dataframe...")

final_df = pd.DataFrame({
    'timestamp': df['timestamp'],
    'predicted_load': predicted_load_full,
    'actual_load': df['load_kW'],
    'anomaly_label': anomaly_labels,
    'risk_score': risk_score
})

# Fill NaN predicted loads with actual loads for initial period
final_df['predicted_load'].fillna(final_df['actual_load'], inplace=True)

print("✓ Final dataframe created")
print(f"\nFinal DataFrame Summary:")
print(final_df.describe())
print(f"\nFirst few rows:")
print(final_df.head(10))

# Save to CSV
final_df.to_csv('transformer_analysis_results.csv', index=False)
print("\n✓ Results saved to 'transformer_analysis_results.csv'")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Predicted vs Actual Load
ax1 = plt.subplot(3, 1, 1)
# Plot only where we have predictions (skip NaN values)
valid_idx = ~np.isnan(predicted_load_full)
ax1.plot(df.loc[valid_idx, 'timestamp'], df.loc[valid_idx, 'load_kW'], 
         label='Actual Load', alpha=0.7, linewidth=1.5)
ax1.plot(df.loc[valid_idx, 'timestamp'], predicted_load_full[valid_idx], 
         label='Predicted Load', alpha=0.7, linewidth=1.5, linestyle='--')
ax1.set_xlabel('Timestamp', fontsize=12)
ax1.set_ylabel('Load (kW)', fontsize=12)
ax1.set_title('Predicted vs Actual Load', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Anomalies highlighted on load curve
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['timestamp'], df['load_kW'], label='Load', alpha=0.6, linewidth=1.5, color='blue')
# Highlight anomalies
anomaly_times = df.loc[anomaly_labels == 1, 'timestamp']
anomaly_loads = df.loc[anomaly_labels == 1, 'load_kW']
ax2.scatter(anomaly_times, anomaly_loads, color='red', s=50, 
           label=f'Anomalies ({anomaly_labels.sum()})', zorder=5, alpha=0.8)
ax2.set_xlabel('Timestamp', fontsize=12)
ax2.set_ylabel('Load (kW)', fontsize=12)
ax2.set_title('Load Curve with Anomalies Highlighted', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Risk Score Timeline
ax3 = plt.subplot(3, 1, 3)
ax3.plot(df['timestamp'], risk_score, linewidth=2, color='orange', alpha=0.8)
ax3.fill_between(df['timestamp'], 0, risk_score, alpha=0.3, color='orange')
ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='High Risk Threshold (0.7)')
ax3.set_xlabel('Timestamp', fontsize=12)
ax3.set_ylabel('Risk Score', fontsize=12)
ax3.set_title('Transformer Risk Score Timeline', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transformer_analysis_plots.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to 'transformer_analysis_plots.png'")

# Additional detailed plot: Test set predictions
fig2, ax = plt.subplots(figsize=(15, 6))
test_indices = range(len(actual_load))
ax.plot(test_indices, actual_load, label='Actual Load', linewidth=2, alpha=0.8)
ax.plot(test_indices, predicted_load, label='Predicted Load', linewidth=2, 
        linestyle='--', alpha=0.8)
ax.set_xlabel('Test Sample Index', fontsize=12)
ax.set_ylabel('Load (kW)', fontsize=12)
ax.set_title('LSTM Model: Predicted vs Actual Load (Test Set)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_test_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Test set predictions saved to 'lstm_test_predictions.png'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nSummary Statistics:")
print(f"  Total samples: {len(final_df)}")
print(f"  Anomalies detected: {anomaly_labels.sum()} ({100*anomaly_labels.sum()/len(anomaly_labels):.2f}%)")
print(f"  Average risk score: {risk_score.mean():.3f}")
print(f"  Max risk score: {risk_score.max():.3f}")
print(f"  High risk periods (>0.7): {(risk_score > 0.7).sum()}")
print(f"\nFiles generated:")
print(f"  - transformer_analysis_results.csv")
print(f"  - transformer_analysis_plots.png")
print(f"  - lstm_test_predictions.png")

plt.show()

