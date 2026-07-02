import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from skl2onnx import to_onnx

def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    
    # We have 5 features:
    # 0: flow_duration (microseconds)
    # 1: fwd_pkts (count)
    # 2: bwd_pkts (count)
    # 3: pkt_size_var (variance of packet sizes in bytes)
    # 4: iat_mean (mean inter-arrival time in microseconds)
    
    half_samples = n_samples // 2
    
    # --- Class 0: Normal Traffic ---
    # Normal flows: moderate duration, moderate packet counts, high packet size variance, moderate IAT
    duration_normal = np.random.uniform(50000, 5000000, half_samples) # 50ms to 5s
    fwd_pkts_normal = np.random.randint(5, 100, half_samples)
    bwd_pkts_normal = np.random.randint(5, 100, half_samples)
    pkt_size_var_normal = np.random.uniform(5000, 150000, half_samples) # varied packet sizes
    iat_mean_normal = duration_normal / (fwd_pkts_normal + bwd_pkts_normal)
    
    X_normal = np.column_stack([
        duration_normal,
        fwd_pkts_normal,
        bwd_pkts_normal,
        pkt_size_var_normal,
        iat_mean_normal
    ])
    y_normal = np.zeros(half_samples, dtype=np.int64)
    
    # --- Class 1: Anomalous Traffic ---
    # Anomalous flows can be DoS (extremely high rate, low size variance) or PortScan (1-2 packets, extremely short)
    n_dos = half_samples // 2
    n_scan = half_samples - n_dos
    
    # DoS/DDoS: short duration, high forward packets, minimal backward packets, uniform small packet sizes, tiny IAT
    duration_dos = np.random.uniform(1000, 50000, n_dos) # 1ms to 50ms
    fwd_pkts_dos = np.random.randint(50, 500, n_dos)
    bwd_pkts_dos = np.random.randint(0, 5, n_dos)
    pkt_size_var_dos = np.random.uniform(0, 100, n_dos) # almost identical packet sizes (e.g. syn flood)
    iat_mean_dos = duration_dos / (fwd_pkts_dos + bwd_pkts_dos + 1e-5)
    
    X_dos = np.column_stack([
        duration_dos,
        fwd_pkts_dos,
        bwd_pkts_dos,
        pkt_size_var_dos,
        iat_mean_dos
    ])
    
    # PortScan: very short duration, 1-2 packets, 0 backward, 0 size variance, 0 IAT
    duration_scan = np.random.uniform(1, 100, n_scan) # <0.1ms
    fwd_pkts_scan = np.ones(n_scan)
    bwd_pkts_scan = np.zeros(n_scan)
    pkt_size_var_scan = np.zeros(n_scan)
    iat_mean_scan = np.zeros(n_scan)
    
    X_scan = np.column_stack([
        duration_scan,
        fwd_pkts_scan,
        bwd_pkts_scan,
        pkt_size_var_scan,
        iat_mean_scan
    ])
    
    X_anomaly = np.vstack([X_dos, X_scan])
    y_anomaly = np.ones(half_samples, dtype=np.int64)
    
    X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
    y = np.concatenate([y_normal, y_anomaly])
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def main():
    print("Generating synthetic network flow dataset...")
    X, y = generate_synthetic_data(5000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Using RandomForestClassifier with max_depth=5 and n_estimators=10 for low latency on edge CPUs
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, preds))
    
    f1 = f1_score(y_test, preds)
    print(f"F1-Score: {f1 * 100:.2f}%")
    
    # Export to ONNX
    print("\nConverting model to ONNX format...")
    # X contains 5 features: flow_duration, fwd_pkts, bwd_pkts, pkt_size_var, iat_mean
    # Input name: "float_input"
    onx = to_onnx(clf, X_train[:1], target_opset=12)
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "nids_model.onnx")
    
    with open(model_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"ONNX model saved successfully to: {model_path}")
    
    # Quick sanity check with onnxruntime in Python
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # Run prediction on first 3 test samples
    test_inputs = X_test[:3]
    pred_ort = sess.run(None, {input_name: test_inputs})
    print("\nONNX Runtime Python Test Prediction:")
    print("Inputs:")
    for inp in test_inputs:
        print(f"  Duration: {inp[0]:.1f}us, Fwd: {int(inp[1])}, Bwd: {int(inp[2])}, Var: {inp[3]:.1f}, IAT: {inp[4]:.1f}us")
    print("Predicted Classes:", pred_ort[0])
    print("Predicted Probabilities:\n", pred_ort[1])

if __name__ == "__main__":
    main()
