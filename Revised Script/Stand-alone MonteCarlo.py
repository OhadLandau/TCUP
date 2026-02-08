"""
Standalone Monte Carlo Gene Importance Script
Loads trained models and runs Monte Carlo simulation without retraining
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
import pickle

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
project_dir = r"C:\Users\oland\PycharmProjects\TissueOfOriginPrediction"
output_dir = os.path.join(project_dir, "MetastaticTOO_IncorpTest", "Full_wFullGenesModifiedMaxPairsv2")
evaluation_dir = os.path.join(output_dir, "TestEvaluation")

# Load data
print("Loading scaled data...")
scaled_data_file = os.path.join(output_dir, "merged_data_withStandardScaler.pkl")
with open(scaled_data_file, 'rb') as f:
    data_scaled = pickle.load(f)

test_data_scaled = data_scaled['test_data_scaled']

# Load feature sets
print("Loading feature sets...")
feature_sets_file = os.path.join(output_dir, "feature_sets.pkl")
with open(feature_sets_file, 'rb') as f:
    feature_sets = pickle.load(f)
feature_columns = feature_sets['FullLasso']
fcols = feature_columns

print(f"Features: {len(fcols)}")
print(f"Test samples: {len(test_data_scaled)}")

# Load models
print("\nLoading trained models...")
snn_model_file = os.path.join(output_dir, 'snn_model.h5')
cae_encoder_file = os.path.join(output_dir, 'cae_encoder.h5')
base_classifiers_file = os.path.join(evaluation_dir, "trained_base_classifiers.pkl")

# Load SNN and extract base network (layer 2 is the shared encoder)
snn_full_model = tf.keras.models.load_model(snn_model_file)
snn_base_network = snn_full_model.layers[2]
print(f"✓ SNN base network extracted (layer 2)")

cae_encoder = tf.keras.models.load_model(cae_encoder_file)
print(f"✓ CAE encoder loaded")

with open(base_classifiers_file, 'rb') as f:
    met_classifiers = pickle.load(f)
print(f"✓ Base classifiers loaded: {list(met_classifiers.keys())}")

# Load best meta-learner
meta_results_file = os.path.join(evaluation_dir, 'meta_learner_restricted_results.csv')
results_df = pd.read_csv(meta_results_file)
results_df_sorted = results_df.sort_values(by='Val_Accuracy', ascending=False)
best_iteration = int(results_df_sorted.iloc[0]['Iteration'])

best_model_path = os.path.join(evaluation_dir, f"best_meta_learner_{best_iteration}.h5")
meta_model = tf.keras.models.load_model(best_model_path)
print(f"✓ Meta-learner loaded from iteration {best_iteration}")
print(f"  Test Accuracy: {results_df_sorted.iloc[0]['Test_Accuracy']:.4f}")

def get_meta_features(classifiers, X):
    meta_feats = []
    for c in classifiers.values():
        meta_feats.append(c.predict_proba(X))
    return np.concatenate(meta_feats, axis=1)

# Prepare test data
print("\nPreparing test data...")
test_data_filtered_df = test_data_scaled.copy().reset_index(drop=True)

# Load label encoder
preprocessed_file = os.path.join(output_dir, "merged_data_preprocessed.pkl")
with open(preprocessed_file, 'rb') as f:
    pre_data = pickle.load(f)
label_encoder = pre_data['label_encoder']

# Generate embeddings and baseline predictions
print("Generating baseline predictions...")
test_features = test_data_filtered_df[fcols].values
test_X_snn = snn_base_network.predict(test_features, verbose=0)
test_X_cae = cae_encoder.predict(test_features, verbose=0)
test_combined_emb = np.concatenate([test_X_snn, test_X_cae], axis=1)
test_meta_feats = get_meta_features(met_classifiers, test_combined_emb)
test_y_pred = meta_model.predict(test_meta_feats, verbose=0).argmax(axis=1)

# Get true labels
y_test_true = label_encoder.transform(test_data_filtered_df['LABEL'].values)

# Baseline accuracy
acc_baseline = accuracy_score(y_test_true, test_y_pred)
print(f"Baseline test accuracy (no noise): {acc_baseline:.4f}")

# Compute per-label baseline accuracies
unique_labels_for_acc = test_data_filtered_df['LABEL'].unique()
baseline_label_acc = {}
for lbl in unique_labels_for_acc:
    idx_ = (test_data_filtered_df['LABEL'] == lbl)
    if np.sum(idx_) == 0:
        continue
    baseline_label_acc[lbl] = accuracy_score(y_test_true[idx_], test_y_pred[idx_])

print(f"Computed baseline accuracies for {len(baseline_label_acc)} labels")

# Monte Carlo simulation
print("\n" + "="*80)
print("STARTING MONTE CARLO GENE IMPORTANCE ANALYSIS")
print("="*80)

num_monte_carlo_cycles = 10
noise_std = 20.0

def get_predictions_after_noise(test_df_noisy):
    """Generate predictions after adding noise to a gene"""
    X_snn_noisy = snn_base_network.predict(test_df_noisy[fcols].values, verbose=0)
    X_cae_noisy = cae_encoder.predict(test_df_noisy[fcols].values, verbose=0)
    combined_emb_noisy = np.concatenate([X_snn_noisy, X_cae_noisy], axis=1)
    meta_feats_noisy = get_meta_features(met_classifiers, combined_emb_noisy)
    y_pred_noisy = meta_model.predict(meta_feats_noisy, verbose=0).argmax(axis=1)
    return y_pred_noisy

list_of_dfs = []

print(f"Running {num_monte_carlo_cycles} Monte Carlo cycles...")
for cycle_idx in range(1, num_monte_carlo_cycles + 1):
    print(f"\n--- Cycle {cycle_idx}/{num_monte_carlo_cycles} ---")

    gene_importance_results = []

    for gene_idx, gene_name in enumerate(feature_columns):
        print(f"  Processing gene {gene_idx}/{len(feature_columns)}: {gene_name}")

        # Copy test data and add noise to this gene
        test_noisy_df = test_data_filtered_df.copy()
        test_noisy_df[gene_name] += np.random.normal(loc=0.0, scale=noise_std, size=len(test_noisy_df))

        # Get predictions
        y_pred_noisy = get_predictions_after_noise(test_noisy_df)
        new_acc = accuracy_score(y_test_true, y_pred_noisy)
        drop_in_acc = acc_baseline - new_acc

        # Per-label accuracy drops
        label_drop_dict = {}
        for lbl in unique_labels_for_acc:
            idx_lbl = (test_noisy_df['LABEL'] == lbl)
            if np.sum(idx_lbl) == 0:
                continue
            new_acc_lbl = accuracy_score(y_test_true[idx_lbl], y_pred_noisy[idx_lbl])
            drop_in_acc_lbl = baseline_label_acc[lbl] - new_acc_lbl
            label_drop_dict[lbl] = drop_in_acc_lbl

        # Build row
        row_dict = {'Gene': gene_name, 'AccuracyDrop': drop_in_acc}
        for lbl in unique_labels_for_acc:
            row_dict[f"AccuracyDrop_{lbl}"] = label_drop_dict.get(lbl, np.nan)

        gene_importance_results.append(row_dict)

    # Create DataFrame for this cycle
    df_cycle = pd.DataFrame(gene_importance_results)
    df_cycle.sort_values('AccuracyDrop', ascending=False, inplace=True)

    # Rank-based p-value
    all_drops = df_cycle['AccuracyDrop'].values
    ranks_desc = rankdata(-all_drops, method='average')
    p_vals = ranks_desc / (len(all_drops) + 1.0)
    df_cycle['p_value'] = p_vals

    # Save cycle CSV
    cycle_csv_path = os.path.join(evaluation_dir, f"monte_carlo_gene_importance_{cycle_idx}th.csv")
    df_cycle.to_csv(cycle_csv_path, index=False)
    print(f"  Saved: {cycle_csv_path}")

    # Store for averaging
    df_cycle_sorted = df_cycle.sort_values('Gene').reset_index(drop=True)
    list_of_dfs.append(df_cycle_sorted)

# Average across cycles
print("\nAveraging results across all Monte Carlo cycles...")
df_final = list_of_dfs[0].copy()
num_cycles = len(list_of_dfs)

for col in df_final.columns:
    if col == 'Gene':
        continue
    for c_idx in range(1, num_cycles):
        df_final[col] += list_of_dfs[c_idx][col]
    df_final[col] /= float(num_cycles)

# Recompute p-value
all_avg_drops = df_final['AccuracyDrop'].values
ranks_desc = rankdata(-all_avg_drops, method='average')
p_vals = ranks_desc / (len(all_avg_drops) + 1.0)
df_final['p_value'] = p_vals

# Sort and save
df_final.sort_values('AccuracyDrop', ascending=False, inplace=True)
final_csv_path = os.path.join(evaluation_dir, "monte_carlo_gene_importance_aggregated.csv")
df_final.to_csv(final_csv_path, index=False)
print(f"\n✅ Final aggregated CSV saved: {final_csv_path}")

# Plot top 20
import matplotlib.pyplot as plt
top_k = 20
df_top_20 = df_final.head(top_k)
plt.figure(figsize=(10, 6))
plt.barh(df_top_20['Gene'][::-1], df_top_20['AccuracyDrop'][::-1], color='red', alpha=0.6)
plt.xlabel("Average Drop in Accuracy (Over Monte Carlo Cycles)")
plt.title(f"Top {top_k} Genes by Avg Accuracy Drop (Noise={noise_std}, n={num_monte_carlo_cycles})")
plt.tight_layout()
plot_path = os.path.join(evaluation_dir, "monte_carlo_gene_importance_top20_aggregated.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✅ Plot saved: {plot_path}")

print("\n" + "="*80)
print("✅ MONTE CARLO COMPLETE!")
print("="*80)

