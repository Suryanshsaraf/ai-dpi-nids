import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Replicate the synthetic NIDS dataset distributions
np.random.seed(42)
N = 2000

# Normal traffic features (duration, fwd_pkts, bwd_pkts, size_var, iat_mean)
duration_norm = np.random.exponential(scale=1500000.0, size=N)
fwd_norm = np.random.randint(5, 100, size=N)
bwd_norm = np.random.randint(5, 100, size=N)
var_norm = np.random.exponential(scale=150000.0, size=N)
iat_norm = duration_norm / (fwd_norm + bwd_norm)

# Anomalies (e.g., fast port scanning: ultra-short duration, 1-2 packets, 0 variance, 0 IAT)
duration_anom = np.random.exponential(scale=100.0, size=N)
fwd_anom = np.random.choice([1, 2], size=N)
bwd_anom = np.zeros(N)
var_anom = np.zeros(N)
iat_anom = np.zeros(N)

X_normal = np.column_stack((duration_norm, fwd_norm, bwd_norm, var_norm, iat_norm))
X_anomaly = np.column_stack((duration_anom, fwd_anom, bwd_anom, var_anom, iat_anom))

X = np.vstack((X_normal, X_anomaly))
y = np.hstack((np.zeros(N), np.ones(N)))

feature_names = ["Flow Duration (us)", "Fwd Packets", "Bwd Packets", "Packet Size Variance", "IAT Mean (us)"]

# Fit RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

# Print Feature Importance Metrics
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n" + "="*45)
print("     AI & DS FEATURE IMPORTANCE REPORT     ")
print("="*45)
for f in range(X.shape[1]):
    print(f" {f+1:2d}. {feature_names[indices[f]]:<25} : {importances[indices[f]]*100:6.2f}%")
print("="*45)

# Generate and save clean Feature Importance bar chart
plt.figure(figsize=(10, 6))
colors = ['#1cc88a', '#4e73df', '#36b9cc', '#f6c23e', '#e74a3b']
plt.bar(range(X.shape[1]), importances[indices], color=[colors[i % len(colors)] for i in range(len(indices))], edgecolor='none', width=0.6)
plt.title("Feature Importance - AI-DPI NIDS Model", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Relative Importance Score", fontsize=12)
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=15, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Styling
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("model/feature_importance.png", dpi=150)
print("\n[AI-DPI] Feature importance plot saved successfully to: model/feature_importance.png\n")
