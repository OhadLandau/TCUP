import os
import unittest
import pandas as pd
import numpy as np
import random
import pickle
import glob
import warnings
import logging
import time
from contextlib import contextmanager
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, GroupShuffleSplit, \
    GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from itertools import combinations
from collections import defaultdict, Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# ============================================================================
# TIMING CONTEXT MANAGER FOR PERFORMANCE BENCHMARKING
# ============================================================================
@contextmanager
def timer(name):
    """
    Context manager for timing code blocks and logging execution time.

    Provides timing information for computational cost analysis and performance
    benchmarking. Logs start time, completion time, and duration in both seconds
    and minutes.

    Args:
        name (str): Descriptive name for the code block being timed

    Yields:
        None: Context manager yields control to the code block

    Example:
        >>> with timer("Data Preprocessing"):
        ...     # preprocessing code ...
        Starting: Data Preprocessing
        Completed Data Preprocessing in 193.42s (3.2m)
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Completed {name} in {elapsed:.2f}s ({elapsed / 60:.1f}m)")


_BASE = 10  # master knob (was 20)
_TICKS = _BASE - 7
_LEGEND = _BASE - 5

plt.rcParams.update({
    'font.size': _BASE,
    'axes.titlesize': _BASE + 2,
    'axes.labelsize': _BASE,
    'xtick.labelsize': _TICKS,
    'ytick.labelsize': _TICKS,
    'legend.fontsize': _LEGEND,
    'figure.titlesize': _BASE + 4,
    'figure.dpi': 300,
})

# Make seaborn honour the same scale
sns.set_context("notebook", font_scale=_BASE / 10)

# ============================================================================
# SEED MANAGEMENT FOR FULL REPRODUCIBILITY
# ============================================================================
# Use a single consistent seed (42) across all libraries for reproducibility
SEED = 42

# Set seeds for all random number generators
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Enable TensorFlow determinism for GPU operations
# This ensures CUDA operations are deterministic (may have performance cost)
try:
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"TensorFlow determinism enabled for full reproducibility (seed={SEED})")
except Exception as e:
    print(f"Warning: Could not enable TensorFlow determinism: {e}")
    print("Results may vary slightly between runs on GPU")




try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


current = script_dir
while current and os.path.basename(current) not in ['TissueOfOriginPrediction', '']:
    parent = os.path.dirname(current)
    if parent == current:  # Reached root
        break
    current = parent

# If we found TissueOfOriginPrediction, use it; otherwise use script_dir
if os.path.basename(current) == 'TissueOfOriginPrediction':
    project_dir = current
else:
    # Fallback: assume data files are in the same directory as script or one level up
    project_dir = os.path.dirname(script_dir) if 'output_external_validation' in script_dir else script_dir

patient_data_file = os.path.join(project_dir, "BalancedTCGA_CancerTranscriptomics.csv")
gtex_reads_dir = os.path.join(project_dir, "GTEx Reads")
metastatic_data_file = os.path.join(project_dir, "metastatic_TOO_dataset_tpm.csv")

# Verify data files exist, print helpful error if not
print(f"Project directory: {project_dir}")
if not os.path.exists(patient_data_file):
    print(f"WARNING: Patient data file not found at: {patient_data_file}")
    print(f"Please ensure data files are in: {project_dir}")

output_base_dir = os.path.join(project_dir, "MetastaticTOO_IncorpTest")
os.makedirs(output_base_dir, exist_ok=True)
gtex_processed_dir = os.path.join(project_dir, "GTExProcessed_noZ")
os.makedirs(gtex_processed_dir, exist_ok=True)

new_output_dir = os.path.join(output_base_dir, "Full_wFullGenesModifiedMaxPairsv2")
os.makedirs(new_output_dir, exist_ok=True)
merged_data_preprocessed_file = os.path.join(new_output_dir, "merged_data_preprocessed.pkl")
gtex_processed_file = os.path.join(new_output_dir, "GTEx_processed.pkl")
tcga_processed_file = os.path.join(new_output_dir, "TCGA_processed.pkl")
metastatic_processed_file = os.path.join(new_output_dir, "metastatic_processed.pkl")
feature_sets_file = os.path.join(new_output_dir, "feature_sets.pkl")
scaler_file = os.path.join(new_output_dir, 'BalancedMetastaticStandardScaler.pkl')
merged_data_scaled_file = os.path.join(new_output_dir, "merged_data_withStandardScaler.pkl")

snn_model_file = os.path.join(new_output_dir, 'snn_model.h5')
cae_encoder_file = os.path.join(new_output_dir, 'cae_encoder.h5')
cae_decoder_file = os.path.join(new_output_dir, 'cae_decoder.h5')
evaluation_dir = os.path.join(new_output_dir, "TestEvaluation")
os.makedirs(evaluation_dir, exist_ok=True)

max_pairs = 1000000
max_usage_per_sample_posneg = 5  # each positivity/neg usage
metastatic_pos_neg_count = 30


# ---------- Unit Tests ------------
class SplitDataTests(unittest.TestCase):
    """
    Basic unit tests to confirm data splits and no data leakage
    """

    def setUp(self):
        if os.path.exists(merged_data_preprocessed_file):
            with open(merged_data_preprocessed_file, 'rb') as f:
                self.pre_data = pickle.load(f)
        else:
            self.pre_data = None

    def test_data_exists(self):
        self.assertIsNotNone(self.pre_data,
                             "Preprocessed data not found; cannot run further tests.")

    def test_no_overlap_in_splits(self):
        """
        Ensures no overlap between train/val/test via SAMPLE_ID
        """
        if self.pre_data is not None:
            train_ids = set(self.pre_data['train_data']['SAMPLE_ID'].values)
            val_ids = set(self.pre_data['val_data']['SAMPLE_ID'].values)
            test_ids = set(self.pre_data['test_data']['SAMPLE_ID'].values)
            self.assertTrue(train_ids.isdisjoint(val_ids),
                            "Overlap between train and val (SAMPLE_ID).")
            self.assertTrue(val_ids.isdisjoint(test_ids),
                            "Overlap between val and test (SAMPLE_ID).")
            self.assertTrue(train_ids.isdisjoint(test_ids),
                            "Overlap between train and test (SAMPLE_ID).")

    def test_label_encoder_integrity(self):
        """
        Ensures label encoder is consistent across sets
        """
        if self.pre_data is not None:
            train_labels = self.pre_data['train_data']['LABEL_NUMERIC'].unique()
            val_labels = self.pre_data['val_data']['LABEL_NUMERIC'].unique()
            test_labels = self.pre_data['test_data']['LABEL_NUMERIC'].unique()
            all_labels = set(np.concatenate([train_labels, val_labels, test_labels]))
            self.assertEqual(len(all_labels),
                             len(self.pre_data['label_encoder'].classes_),
                             "Mismatch in label encoder classes")


# ---------- Data Processing ------------
def process_gtex_files(gtex_reads_dir, gtex_processed_dir):
    """
    Load or process GTEx gene expression data from CSV files.

    Processes GTEx RNA-seq data files, extracting tissue type information from
    filenames and organizing data into robust and all-tissue-type categories.
    Implements caching to avoid reprocessing if output files already exist.

    Args:
        gtex_reads_dir (str): Directory containing GTEx CSV files (gene_reads_*.csv)
        gtex_processed_dir (str): Directory to save processed GTEx data files

    Returns:
        tuple: (gtex_robust, gtex_all) where
            - gtex_robust: pd.DataFrame with robust tissue categories (first word of filename)
            - gtex_all: pd.DataFrame with all tissue type categories (full filename)

    Notes:
        - Files are expected to have format: gene_reads_<tissue>_<subtype>.csv
        - Robust tissue uses first word (e.g., "Brain" from "brain_amygdala")
        - All tissue uses full name (e.g., "Brain_amygdala")
        - If processed files exist and are valid, they are loaded instead of reprocessing
    """
    robust_file = os.path.join(gtex_processed_dir, "GTEx_Robust.csv")
    all_types_file = os.path.join(gtex_processed_dir, "GTEx_All_Types.csv")
    if os.path.exists(robust_file) and os.path.exists(all_types_file):
        def load_large_csv_in_chunks(file_path, chunksize=10000):
            chunk_list = []
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                chunk_list.append(chunk)
            return pd.concat(chunk_list, ignore_index=True)

        gtex_robust = load_large_csv_in_chunks(robust_file)
        gtex_all = load_large_csv_in_chunks(all_types_file)
        if 'TISSUE_ROBUST' not in gtex_robust.columns or 'TISSUE_ALL' not in gtex_all.columns:
            os.remove(robust_file)
            os.remove(all_types_file)
            return process_gtex_files(gtex_reads_dir, gtex_processed_dir)
    else:
        gtex_files = glob.glob(os.path.join(gtex_reads_dir, "*.csv"))
        data_list_robust = []
        data_list_all = []
        for file in gtex_files:
            filename = os.path.basename(file)
            base_name = filename.replace("gene_reads_", "").replace(".csv", "")
            tissue_parts = base_name.split('_')
            robust_tissue = tissue_parts[0].capitalize()
            all_tissue = '_'.join(tissue_parts).capitalize()
            df = pd.read_csv(file, index_col=0).T
            df['TISSUE_ROBUST'] = robust_tissue
            df['TISSUE_ALL'] = all_tissue
            data_list_robust.append(df.drop(columns=['TISSUE_ALL']))
            data_list_all.append(df.drop(columns=['TISSUE_ROBUST']))
        gtex_robust = pd.concat(data_list_robust, ignore_index=True)
        gtex_all = pd.concat(data_list_all, ignore_index=True)
        gtex_robust.to_csv(robust_file, index=False)
        gtex_all.to_csv(all_types_file, index=False)
    return gtex_robust, gtex_all


def process_gtex_files_donor_aware(gtex_reads_dir, gtex_processed_file):
    """
    Process GTEx files into a single concatenated dataframe with donor awareness.

    Processes GTEx RNA-seq data files, extracting donor IDs from sample IDs and
    tissue labels from filenames. Concatenates all files along rows (samples),
    preserving donor information for group-based splitting to prevent data leakage.

    Args:
        gtex_reads_dir (str): Directory containing GTEx CSV files (gene_reads_*.csv)
        gtex_processed_file (str): Path to save/load processed GTEx data pickle file

    Returns:
        pd.DataFrame: Concatenated GTEx data with columns:
            - Gene expression columns (all genes from input files)
            - DONOR_ID: Extracted from sample ID (e.g., "GTEX-1192X" from "GTEX-1192X-0011-R5a-SM-DNZZA")
            - LABEL: Tissue label with "_GTEx" suffix (e.g., "Brain_GTEx")
            - SAMPLE_ID: Full GTEx sample identifier

    Notes:
        - Sample IDs format: GTEX-<DONOR>-<SAMPLE>-<BATCH>-<ANALYTE>-<ALIQUOT>
        - Donor ID extracted as first two parts: GTEX-<DONOR>
        - Tissue name extracted from filename: gene_reads_<tissue>_<subtype>.csv -> <Tissue>_GTEx
        - Duplicate gene columns are removed (keeps first occurrence)
        - Implements checkpointing: if processed file exists and is valid, loads it instead
    """
    if os.path.exists(gtex_processed_file):
        try:
            print(f"Loading preprocessed GTEx data from: {gtex_processed_file}")
            with open(gtex_processed_file, 'rb') as f:
                gtex_data = pickle.load(f)
            # Verify the loaded data has required columns
            if 'DONOR_ID' in gtex_data.columns and 'LABEL' in gtex_data.columns and 'SAMPLE_ID' in gtex_data.columns:
                print(f"  Successfully loaded {len(gtex_data):,} GTEx samples")
                return gtex_data
            else:
                print(f"  WARNING: Loaded file missing required columns. Regenerating...")
                os.remove(gtex_processed_file)
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"  ERROR: Failed to load GTEx processed file ({e}). Regenerating...")
            if os.path.exists(gtex_processed_file):
                os.remove(gtex_processed_file)

    print("Processing GTEx files and concatenating into single dataframe...")
    gtex_files = glob.glob(os.path.join(gtex_reads_dir, "gene_reads_*.csv"))
    print(f"Found {len(gtex_files)} GTEx CSV files")

    data_list = []
    for file in gtex_files:
        filename = os.path.basename(file)
        # Extract tissue name: gene_reads_brain_amygdala.csv -> Brain
        base_name = filename.replace("gene_reads_", "").replace(".csv", "")
        tissue_parts = base_name.split('_')
        tissue_name = tissue_parts[0].capitalize()  # brain -> Brain

        # Read CSV: genes as index, transpose so samples are rows
        df = pd.read_csv(file, index_col=0).T

        # Extract donor IDs from sample IDs (row index after transpose)
        # Format: GTEX-1192X-0011-R5a-SM-DNZZA -> GTEX-1192X
        donor_ids = []
        labels = []
        sample_ids = []

        for sample_id in df.index:
            if sample_id.startswith('GTEX-'):
                parts = sample_id.split('-')
                if len(parts) >= 2:
                    donor_id = f"{parts[0]}-{parts[1]}"  # GTEX-1192X
                    label = f"{tissue_name}_GTEx"  # Brain_GTEx
                else:
                    donor_id = sample_id
                    label = f"{tissue_name}_GTEx"
            else:
                # Skip non-GTEx rows (shouldn't happen, but safety check)
                continue

            donor_ids.append(donor_id)
            labels.append(label)
            sample_ids.append(sample_id)

        # Filter dataframe to only GTEx samples
        df_filtered = df.loc[sample_ids].copy()
        df_filtered['DONOR_ID'] = donor_ids
        df_filtered['LABEL'] = labels
        df_filtered['SAMPLE_ID'] = sample_ids

        data_list.append(df_filtered)
        print(f"  Processed {filename}: {len(df_filtered)} samples")

    # Concatenate all files along ROWS (axis=0) - each file adds more samples, not more columns
    gtex_data = pd.concat(data_list, ignore_index=True, axis=0)  # axis=0 explicitly (rows)

    print(f"After concatenation: {len(gtex_data)} rows (samples), {len(gtex_data.columns)} columns (genes + metadata)")

    # Check and remove duplicate columns (keep first occurrence)
    if gtex_data.columns.duplicated().any():
        print(f"WARNING: GTEx concatenation found {gtex_data.columns.duplicated().sum()} duplicate column names!")
        print("Duplicate columns:", gtex_data.columns[gtex_data.columns.duplicated()].tolist())
        gtex_data = gtex_data.loc[:, ~gtex_data.columns.duplicated()]
        print("Removed duplicate columns after concatenation.")
        print(f"After cleanup: {len(gtex_data)} rows (samples), {len(gtex_data.columns)} columns")

    print(f"Total GTEx samples after concatenation: {len(gtex_data):,}")

    # Save processed GTEx data
    with open(gtex_processed_file, 'wb') as f:
        pickle.dump(gtex_data, f)
    print(f"GTEx data saved to: {gtex_processed_file}")

    return gtex_data


# ---------- Pair Creation Logic ------------
def create_pairs_by_logic(data,
                          max_pairs=40000,
                          max_usage_per_labelpos=max_usage_per_sample_posneg,
                          max_usage_per_labelneg=max_usage_per_sample_posneg,
                          overrepresent_metastatic=False):
    """
    Create positive and negative sample pairs for Siamese network training.

    Implements pair generation strategy for contrastive learning. Positive pairs
    are created from samples with the same tissue label, while negative pairs
    are created from samples with different tissue labels. This enables the
    Siamese network to learn discriminative embeddings by contrasting similar
    and dissimilar samples.

    To address class imbalance, metastatic samples can be overrepresented by
    increasing their participation in pairs beyond the standard limit. This helps
    the model learn metastatic tissue patterns more effectively.

    Args:
        data (pd.DataFrame): Input data with columns 'LABEL', 'SOURCE', and gene features.
            Must have index that can be used to reference samples.
        max_pairs (int): Maximum total pairs to generate. Default: 40000.
            This serves as a memory limit to prevent excessive pair generation.
        max_usage_per_labelpos (int): Maximum times a sample can appear in positive pairs.
            Prevents any single sample from dominating positive pair generation.
            Default: 5.
        max_usage_per_labelneg (int): Maximum times a sample can appear in negative pairs.
            Prevents any single sample from dominating negative pair generation.
            Default: 5.
        overrepresent_metastatic (bool): If True, increase metastatic sample participation
            to 'metastatic_pos_neg_count' appearances (default: 30) for both positive
            and negative pairs. This helps address class imbalance. Default: False.

    Returns:
        tuple: (pairs, labels) where
            - pairs: np.array of shape (n_pairs, 2) with sample indices for each pair
            - labels: np.array of shape (n_pairs,) with 1=positive pair, 0=negative pair

    Notes:
        - Positive pairs are created first (within-label), then negative (across-label)
        - Usage limits prevent any single sample from dominating the training set
        - Overrepresentation helps the model learn metastatic tissue patterns
        - Pairs are shuffled after generation to prevent ordering bias
        - Random shuffling uses fixed seed for reproducibility
    """
    data = data.reset_index(drop=True)
    positive_pairs = []
    negative_pairs = []
    usage_counts_pos = defaultdict(int)
    usage_counts_neg = defaultdict(int)

    label_indices = data.groupby('LABEL').indices
    labels = list(label_indices.keys())
    metastatic_indices = data[data['SOURCE'] == 'METASTATIC'].index.tolist()
    non_metastatic_indices = data[data['SOURCE'] != 'METASTATIC'].index.tolist()

    # 1) Positive pairs
    for label in labels:
        indices_lbl = list(label_indices[label])
        np.random.RandomState(SEED).shuffle(indices_lbl)
        for i in range(len(indices_lbl)):
            for j in range(i + 1, len(indices_lbl)):
                if (usage_counts_pos[indices_lbl[i]] < max_usage_per_labelpos and
                        usage_counts_pos[indices_lbl[j]] < max_usage_per_labelpos):
                    positive_pairs.append((indices_lbl[i], indices_lbl[j]))
                    usage_counts_pos[indices_lbl[i]] += 1
                    usage_counts_pos[indices_lbl[j]] += 1
                    if len(positive_pairs) >= max_pairs // 2:
                        break
            if len(positive_pairs) >= max_pairs // 2:
                break
        if len(positive_pairs) >= max_pairs // 2:
            break

    # 2) Negative pairs
    label_combos = list(combinations(labels, 2))
    for (l1, l2) in label_combos:
        idx1 = list(label_indices[l1])
        idx2 = list(label_indices[l2])
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx1)
        rng.shuffle(idx2)
        for a in idx1:
            for b in idx2:
                if (usage_counts_neg[a] < max_usage_per_labelneg and
                        usage_counts_neg[b] < max_usage_per_labelneg):
                    negative_pairs.append((a, b))
                    usage_counts_neg[a] += 1
                    usage_counts_neg[b] += 1
                    if len(negative_pairs) >= max_pairs // 2:
                        break
            if len(negative_pairs) >= max_pairs // 2:
                break
        if len(negative_pairs) >= max_pairs // 2:
            break

    # 3) Overrepresent metastatic if requested
    if overrepresent_metastatic:
        meta_counts_pos = defaultdict(int)
        meta_pos = []
        for i in range(len(metastatic_indices)):
            for j in range(i + 1, len(metastatic_indices)):
                if (meta_counts_pos[metastatic_indices[i]] < metastatic_pos_neg_count and
                        meta_counts_pos[metastatic_indices[j]] < metastatic_pos_neg_count):
                    meta_pos.append((metastatic_indices[i], metastatic_indices[j]))
                    meta_counts_pos[metastatic_indices[i]] += 1
                    meta_counts_pos[metastatic_indices[j]] += 1

        meta_neg = []
        for m in metastatic_indices:
            cnt_neg = 0
            for n in non_metastatic_indices:
                if cnt_neg >= metastatic_pos_neg_count:
                    break
                meta_neg.append((m, n))
                cnt_neg += 1

        for p in meta_pos:
            if p not in positive_pairs:
                positive_pairs.append(p)
        for p in meta_neg:
            if p not in negative_pairs:
                negative_pairs.append(p)

    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    all_pairs, all_labels = shuffle(all_pairs, all_labels, random_state=42)
    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]
        all_labels = all_labels[:max_pairs]

    return np.array(all_pairs), np.array(all_labels)


# ---------- PairGenerator ------------
class PairGenerator(Sequence):
    """
    Keras Sequence generator for Siamese network pair-based training.

    Generates batches of sample pairs (X1, X2) with corresponding labels (1=similar, 0=dissimilar)
    for training or evaluating Siamese neural networks. Supports optional data augmentation
    and can work with subset indices for train/validation splits.

    Attributes:
        pairs (np.array): Array of shape (n_pairs, 2) with sample indices for each pair
        labels (np.array): Array of shape (n_pairs,) with binary labels (1=positive, 0=negative)
        data (pd.DataFrame): Full dataset containing samples and features
        feature_columns (list): List of column names to use as features
        batch_size (int): Number of pairs per batch
        augment (bool): Whether to apply data augmentation (additive Gaussian noise)
        indices (np.array, optional): Subset indices to use (for train/val splits)

    Methods:
        __len__(): Returns number of batches
        __getitem__(idx): Returns batch of (X1, X2) pairs and labels

    Notes:
        - Implements Keras Sequence interface for efficient data loading
        - Supports optional augmentation via Gaussian noise (std=0.01)
        - Pairs are indexed into the full dataset to extract feature vectors
    """

    def __init__(self,
                 pairs,
                 labels,
                 data,
                 feature_columns,
                 batch_size=64,
                 augment=False,
                 indices=None):
        if indices is not None:
            self.pairs = pairs[indices]
            self.labels = labels[indices]
        else:
            self.pairs = pairs
            self.labels = labels
        self.data = data.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.indices = np.arange(len(self.pairs))
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_pairs = self.pairs[batch_indices]
        batch_labels = self.labels[batch_indices]
        X1_batch = self.data.iloc[batch_pairs[:, 0]][self.feature_columns].values.astype(np.float32)
        X2_batch = self.data.iloc[batch_pairs[:, 1]][self.feature_columns].values.astype(np.float32)
        if self.augment:
            X1_batch += np.random.normal(0, 0.01, X1_batch.shape)
            X2_batch += np.random.normal(0, 0.01, X2_batch.shape)
        return [X1_batch, X2_batch], batch_labels


# ---------- Model Builders ------------
def create_modified_contrastive_autoencoder(input_shape):
    """
    Create a contrastive autoencoder (CAE) with L2-normalized latent space.

    Builds an autoencoder architecture with encoder-decoder structure where the
    latent representation is L2-normalized. This normalization enables the latent
    space to be used for contrastive learning and similarity computations.

    Architecture:
        - Encoder: input -> 256 -> 128 -> 64 (L2-normalized)
        - Decoder: 64 -> 128 -> 256 -> input_shape
        - All layers use ReLU activation except latent (tanh) and output (linear)
        - Batch normalization and dropout (0.5) applied throughout
        - L2 regularization (1e-4) on all dense layers

    Args:
        input_shape (int): Number of input features (gene expression dimensions)

    Returns:
        tuple: (encoder, decoder, autoencoder) where
            - encoder: Keras Model mapping input -> 64-dim L2-normalized latent
            - decoder: Keras Model mapping 64-dim latent -> reconstructed input
            - autoencoder: Keras Model combining encoder and decoder end-to-end

    Notes:
        - Latent space is L2-normalized to unit sphere for contrastive learning
        - Uses He normal initialization with fixed seed for reproducibility
        - Model weights are initialized deterministically based on global SEED
    """
    initializer = tf.keras.initializers.HeNormal(seed=SEED)

    # Encoder
    input_layer = layers.Input(shape=(input_shape,))
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    encoded = layers.Dense(64, activation='tanh', kernel_initializer=initializer,
                           kernel_regularizer=regularizers.l2(1e-4),
                           name='encoded_layer')(x)
    encoded = layers.Lambda(lambda xx: K.l2_normalize(xx, axis=1))(encoded)
    encoder = models.Model(input_layer, encoded, name='encoder')

    # Decoder
    encoded_input = layers.Input(shape=(64,))
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(encoded_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_shape, activation='linear',
                           kernel_initializer=initializer)(x)
    decoder = models.Model(encoded_input, decoded, name='decoder')

    # Combine
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = models.Model(input_layer, autoencoder_output, name='autoencoder')

    return encoder, decoder, autoencoder


def create_modified_siamese_network(input_shape):
    """
    Create a base multi-layer perceptron (MLP) network for Siamese neural network.

    Builds a feature extraction network that maps input gene expression vectors
    to a 64-dimensional embedding space. This base network is used in pairs within
    the Siamese architecture to learn discriminative representations.

    Architecture:
        - Input -> 256 -> 128 -> 64
        - All layers use ReLU activation
        - Batch normalization after each dense layer
        - Dropout (0.5) after first two layers
        - L2 regularization (1e-5) on all dense layers

    Args:
        input_shape (int): Number of input features (gene expression dimensions)

    Returns:
        tf.keras.Model: Base network mapping input -> 64-dim embedding

    Notes:
        - Used as shared weights in Siamese network for pair-based learning
        - Uses He normal initialization with fixed seed for reproducibility
        - Output embeddings are used for similarity computation between pairs
    """
    initializer = tf.keras.initializers.HeNormal(seed=SEED)
    input_x = layers.Input(shape=(input_shape,))
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5))(input_x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    base_network = models.Model(input_x, x)
    return base_network


def get_hard_examples_snn(model, data_generator,
                          top_k_percent=0.2,
                          min_num_samples=None,
                          mode='uncertainty'):
    """
    Identify hard examples for Siamese network training using hardness scoring.

    Implements hard example mining to select pairs that are difficult for the model
    to classify correctly. Hardness is computed as a combination of binary cross-entropy
    loss and a confidence measure, enabling focus on either uncertain or overconfident
    predictions depending on the mode.

    Args:
        model (tf.keras.Model): Trained Siamese network model
        data_generator (PairGenerator): Data generator providing pairs and labels
        top_k_percent (float): Fraction of hardest examples to select (0.0-1.0).
            Default: 0.2 (top 20%)
        min_num_samples (int, optional): Minimum number of samples to return.
            If None, uses top_k_percent only. Default: None
        mode (str): Hardness computation mode. Options:
            - 'uncertainty': Selects uncertain predictions (hardness = BCE + (1 - |pred-0.5|))
            - 'overconfident': Selects overconfident wrong predictions (hardness = BCE + |pred-0.5|)
            Default: 'uncertainty'

    Returns:
        np.array: Indices of hard examples in the data generator

    Notes:
        - Hardness score combines prediction error (BCE) with confidence measure
        - Uncertainty mode prioritizes examples where model is unsure (pred ≈ 0.5)
        - Overconfident mode prioritizes examples where model is confidently wrong
        - Useful for curriculum learning or hard negative mining strategies
    """
    import math

    def single_sample_bce(y_true, y_pred, eps=1e-7):
        y_clamped = max(min(float(y_pred), 1 - eps), eps)
        return - (float(y_true) * math.log(y_clamped) +
                  (1.0 - float(y_true)) * math.log(1 - y_clamped))

    hardness_values = []
    indices = []

    for batch_idx in range(len(data_generator)):
        (X1_batch, X2_batch), y_batch = data_generator[batch_idx]
        y_pred = model.predict([X1_batch, X2_batch], verbose=0).ravel()
        start_idx = batch_idx * data_generator.batch_size
        end_idx = start_idx + len(y_batch)
        batch_ind = data_generator.indices[start_idx:end_idx]

        for i in range(len(y_batch)):
            lbl = float(y_batch[i])
            pp = float(y_pred[i])
            bce = single_sample_bce(lbl, pp)
            if mode.lower() == 'uncertainty':
                measure = 1.0 - abs(pp - 0.5)
            else:
                measure = abs(pp - 0.5)
            hardness = bce + measure
            hardness_values.append(hardness)

        indices.extend(batch_ind)

    hardness_values = np.array(hardness_values)
    indices = np.array(indices)
    sorted_desc = np.argsort(hardness_values)[::-1]
    num_samples = int(top_k_percent * len(hardness_values))
    if min_num_samples is not None:
        num_samples = max(num_samples, min_num_samples)
    num_samples = min(num_samples, len(hardness_values))

    hard_inds = sorted_desc[:num_samples]
    if mode.lower() == 'overconfident':
        print(f"Number of 'High Overconfidence + High Loss' samples: {len(hard_inds)}")

    return indices[hard_inds]


def get_hard_examples_cae(model, data_generator,
                          top_k_percent=0.2,
                          min_num_samples=None,
                          margin=1.0,
                          mode='uncertainty'):
    """
    Identify top-k% 'hard' CAE examples by summing:
      - Single-sample contrastive loss (on distz)
      - Single-sample MSE for X1 reconstruction, X2 reconstruction
      - Plus a "measure":
          * uncertainty => 1.0 - |dist - 0.5|
          * overconfident => |dist - 0.5|

    The generator returns:
      ([X1_batch, X2_batch],
       [y_contrast, X1_original, X2_original])

    The model outputs => [distz_batch, decA_batch, decB_batch].
    """

    import numpy as np

    def single_sample_contrastive_loss(y_true, dist, margin=1.0):
        """If label=1 => dist^2, else => max(margin-dist,0)^2."""
        if y_true == 1.0:
            return dist ** 2
        else:
            return max(margin - dist, 0.0) ** 2

    def single_sample_mse(x_true, x_dec):
        """Mean squared error for one sample reconstruction."""
        return np.mean((x_true - x_dec) ** 2)

    hardness_values = []
    indices = []

    for batch_idx in range(len(data_generator)):
        # Our generator yields: X=(X1_batch,X2_batch), y=(y_contrast,X1_orig,X2_orig)
        (X1_batch, X2_batch), y_batch = data_generator[batch_idx]
        y_contrast_batch = y_batch[0]  # shape (batch_size,1) typically
        X1_orig_batch = y_batch[1]  # shape (batch_size, input_dim)
        X2_orig_batch = y_batch[2]  # shape (batch_size, input_dim)

        # Model outputs 3 arrays: distz, decA, decB
        distz_batch, decA_batch, decB_batch = model.predict([X1_batch, X2_batch], verbose=0)

        # Identify indices in the entire dataset
        start_idx = batch_idx * data_generator.batch_size
        end_idx = start_idx + len(X1_batch)
        batch_indices = data_generator.indices[start_idx:end_idx]

        # Compute hardness for each sample in the batch
        for i in range(len(X1_batch)):
            # The contrastive label (0 or 1)
            lbl = float(y_contrast_batch[i, 0])  # or [i] if shape=(batch_size,)
            dist_val = float(distz_batch[i, 0])

            # Reconstructions
            x1_dec = decA_batch[i]
            x2_dec = decB_batch[i]

            # Original X1, X2
            x1_true = X1_orig_batch[i]
            x2_true = X2_orig_batch[i]

            # 1) Contrastive loss
            c_loss = single_sample_contrastive_loss(lbl, dist_val, margin=margin)

            # 2) Reconstruction MSE
            mseA = single_sample_mse(x1_true, x1_dec)
            mseB = single_sample_mse(x2_true, x2_dec)
            sample_loss = c_loss + mseA + mseB

            # 3) Uncertainty or Overconfident measure
            if mode.lower() == 'uncertainty':
                measure = 1.0 - abs(dist_val - 0.5)
            else:  # 'overconfident'
                measure = abs(dist_val - 0.5)

            hardness = sample_loss + measure
            hardness_values.append(hardness)

        # Track these sample indices
        indices.extend(batch_indices)

    hardness_values = np.array(hardness_values, dtype=float)
    indices = np.array(indices)

    # Sort hardness descending
    sorted_desc = np.argsort(hardness_values)[::-1]

    # Pick top_k%
    num_samples = int(top_k_percent * len(sorted_desc))
    if min_num_samples is not None:
        num_samples = max(num_samples, min_num_samples)
    num_samples = min(num_samples, len(sorted_desc))

    return indices[sorted_desc[:num_samples]]


def plot_confusion_matrix(labels_true, labels_pred, classes, title, filename,
                          normalize=False, no_numbers=False):
    """
    Draws a confusion matrix.  Font sizes come from rcParams, figure size
    adapts to the number of classes so long labels never collide.
    """
    cm = confusion_matrix(labels_true, labels_pred, labels=classes)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # dynamic figure size (0.55 inch per label, min 6×5)
    w = max(6, 0.55 * len(classes))
    h = max(5, 0.45 * len(classes))
    plt.figure(figsize=(w, h))

    sns.heatmap(
        cm,
        annot=None if (no_numbers and normalize) else True,
        fmt='.2f' if normalize else 'd',
        xticklabels=classes,
        yticklabels=classes,
        cmap='RdBu_r' if normalize else 'Blues',
        vmin=0 if normalize else None,
        vmax=1 if normalize else None,
        annot_kws={'fontsize': plt.rcParams['font.size']}  # <- matches global
    )

    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, classes, title, filename):
    y_true_binarized = to_categorical(y_true, num_classes=len(classes))
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("hls", len(classes))
    for i in range(len(classes)):
        if np.sum(y_true_binarized[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label='{0} (AUC={1:0.2f})'.format(classes[i], roc_auc_val))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),
               ncol=4, fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ---------- Meta-Learner ------------
def create_meta_learner(h1=128, h2=128, h3=128, dropout_rate=0.5,
                        l2_reg=1e-5, optimizer='adam',
                        input_dim=128, num_classes=10):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(h1, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(h2, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(h3, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def get_meta_features(classifiers, X):
    meta_feats = []
    for c in classifiers.values():
        meta_feats.append(c.predict_proba(X))
    return np.concatenate(meta_feats, axis=1)


def get_meta_features_cv(classifiers, X, y, cv_folds=5, random_state=SEED):
    """
    Generate meta-features using cross-validation to avoid data leakage.
    For each fold, train classifiers on training fold and predict on validation fold.
    This ensures meta-features for training data don't leak information.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    meta_feats = np.zeros((len(X), sum(len(c.classes_) for c in classifiers.values())))

    fold_idx = 0
    for train_idx, val_idx in skf.split(X, y):
        fold_idx += 1
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y[train_idx]

        # Train classifiers on this fold's training data
        fold_classifiers = {}
        for c_name, base_clf in classifiers.items():
            # Create a fresh copy of the classifier
            if c_name == 'LogisticRegression':
                fold_clf = LogisticRegression(random_state=random_state)
            elif c_name == 'KNN':
                fold_clf = KNeighborsClassifier()
            elif c_name == 'RandomForest':
                fold_clf = RandomForestClassifier(random_state=random_state)
            elif c_name == 'SVM':
                fold_clf = SVC(probability=True, random_state=random_state)
            elif c_name == 'XGBoost':
                fold_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                         verbosity=0, objective='multi:softprob', random_state=random_state)
                fold_clf.set_params(num_class=len(np.unique(y)))
            else:
                fold_clf = base_clf.__class__(**base_clf.get_params())

            fold_clf.fit(X_train_fold, y_train_fold)
            fold_classifiers[c_name] = fold_clf

        # Generate meta-features for validation fold
        fold_meta_feats = get_meta_features(fold_classifiers, X_val_fold)
        meta_feats[val_idx] = fold_meta_feats

    return meta_feats


def plot_train_val_loss_two_blocks(history,
                                   uncertainty_start,
                                   overconfidence_start,
                                   end_hem,
                                   plot_title,
                                   plot_filename):
    """
    Train / Val loss curve with three vertical markers.
    Legend font-size now follows axes.title size → clearly readable.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if 'loss' not in history.history:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, 'No history available.',
                 ha='center', va='center',
                 fontsize=mpl.rcParams['axes.labelsize'])
        plt.title(plot_title)
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        return

    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, 'r-', label='Val Loss')

    # vertical markers
    plt.axvline(x=uncertainty_start + 0.5, color='red', ls=':', label='Uncertainty HEM')
    plt.axvline(x=overconfidence_start + 0.5, color='gold', ls='--', lw=2, label='Overconfidence HEM')
    plt.axvline(x=end_hem + 0.5, color='black', ls=':', label='All HEM End')

    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # <<<  larger legend
    big_font = mpl.rcParams['axes.titlesize']
    plt.legend(loc='upper right', fontsize=big_font)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def evaluate_cae_autoencoder(cae_encoder, cae_decoder, data,
                             feature_columns, output_dir):
    """
    1) Compute reconstruction for ALL rows in 'data'.
    2) Pick top-2 with the lowest MSE, ensuring they come from different LABELs.
    3) Plot Original(Blue) vs Reconstructed(Red),
       Title => "<LABEL>_<SOURCE>" (no double-suffix).
    """

    if len(data) == 0:
        return

    X_all = data[feature_columns].values
    X_enc_all = cae_encoder.predict(X_all)
    X_dec_all = cae_decoder.predict(X_enc_all)

    # Compute MSE for each sample
    mse_vals = np.mean((X_all - X_dec_all) ** 2, axis=1)
    sorted_idx = np.argsort(mse_vals)

    # Pick first sample as the absolute best MSE
    best_idx1 = sorted_idx[0]
    lbl1 = data.loc[data.index[best_idx1], 'LABEL']

    # Then pick the next best from a different LABEL
    best_idx2 = None
    for idx_ in sorted_idx[1:]:
        if data.loc[data.index[idx_], 'LABEL'] != lbl1:
            best_idx2 = idx_
            break
    # Fallback if every row is the same LABEL
    if best_idx2 is None:
        best_idx2 = sorted_idx[1]

    best_2_idx = [best_idx1, best_idx2]

    fig, axs = plt.subplots(len(best_2_idx), 1, figsize=(10, 6))
    if len(best_2_idx) == 1:
        axs = [axs]

    for i, idx_ in enumerate(best_2_idx):
        label_str = data.loc[data.index[idx_], 'LABEL']
        source_str = data.loc[data.index[idx_], 'SOURCE']

        # Remove any repeated "_SOURCE" to avoid double suffix
        suffix = f"_{source_str}"
        if label_str.endswith(suffix):
            label_str = label_str[:-len(suffix)]

        X_orig = X_all[idx_]
        X_dec = X_dec_all[idx_]

        axs[i].fill_between(range(len(X_orig)), X_orig,
                            color='blue', alpha=0.5, label='Original')
        axs[i].fill_between(range(len(X_dec)), X_dec,
                            color='red', alpha=0.5, label='Reconstructed')

        axs[i].set_title(f"{label_str}_{source_str}")
        axs[i].legend()

    plt.tight_layout()
    recon_path = os.path.join(output_dir, 'cae_reconstruction_top2.png')
    plt.savefig(recon_path, dpi=300)
    plt.close()


# ---------------- MAIN SCRIPT START ----------------
print("Loading and preprocessing data...")
with timer("Full Pipeline Execution"):
    # CHECKPOINT LOGIC: Check for final scaled data FIRST (most complete checkpoint)
    if os.path.exists(merged_data_scaled_file):
        # Final checkpoint exists - load it and skip ALL preprocessing
        print("Final scaled data checkpoint found. Loading and skipping all preprocessing...")
        try:
            with open(merged_data_scaled_file, 'rb') as f:
                data_scaled = pickle.load(f)

            # VALIDATE CHECKPOINT INTEGRITY
            required_keys = ['train_data_scaled', 'val_data_scaled', 'test_data_scaled', 'scaler']
            missing_keys = [k for k in required_keys if k not in data_scaled]
            if missing_keys:
                print(f"ERROR: Checkpoint corrupted. Missing keys: {missing_keys}")
                print("Deleting corrupted checkpoint and regenerating...")
                os.remove(merged_data_scaled_file)
                skip_to_training = False
            else:
                train_data_scaled = data_scaled['train_data_scaled']
                val_data_scaled = data_scaled['val_data_scaled']
                test_data_scaled = data_scaled['test_data_scaled']
                scaler = data_scaled['scaler']

                # VALIDATE DATA SHAPES
                if len(train_data_scaled) == 0 or 'LABEL' not in train_data_scaled.columns:
                    print("ERROR: Checkpoint has invalid data. Regenerating...")
                    os.remove(merged_data_scaled_file)
                    skip_to_training = False
                else:
                    # Load preprocessed data to get metadata (label_encoder, common_genes, feature_columns)
                    if os.path.exists(merged_data_preprocessed_file):
                        with open(merged_data_preprocessed_file, 'rb') as f:
                            pre_data_loaded = pickle.load(f)
                        common_genes = pre_data_loaded['common_genes']
                        label_encoder = pre_data_loaded['label_encoder']
                    else:
                        raise FileNotFoundError(
                            f"ERROR: {merged_data_scaled_file} exists but {merged_data_preprocessed_file} is missing. Cannot load metadata.")

                    # Load feature sets
                    if os.path.exists(feature_sets_file):
                        with open(feature_sets_file, 'rb') as f:
                            feature_sets = pickle.load(f)
                        feature_columns = feature_sets['FullLasso']
                    else:
                        raise FileNotFoundError(
                            f"ERROR: {merged_data_scaled_file} exists but {feature_sets_file} is missing. Cannot load feature columns.")

                    # Define train_data, val_data, test_data from scaled versions for compatibility
                    # (They're the same data, just scaled - needed for code that references them)
                    train_data = train_data_scaled.copy()
                    val_data = val_data_scaled.copy()
                    test_data = test_data_scaled.copy()

                    # CRITICAL FIX: Reconstruct pre_data from scaled data for compatibility
                    # This ensures pre_data exists for code that references it later
                    pre_data = {
                        'merged_data': pd.concat([train_data_scaled, val_data_scaled, test_data_scaled],
                                                 ignore_index=True),
                        'common_genes': common_genes,
                        'train_data': train_data_scaled.copy(),
                        'val_data': val_data_scaled.copy(),
                        'test_data': test_data_scaled.copy(),
                        'label_encoder': label_encoder
                    }


                    # Ensure SOURCE column exists for all paths (checkpoint or fresh)
                    def get_source_from_label(x):
                        if x.endswith('_METASTATIC'):
                            return 'METASTATIC'
                        elif x.endswith('_TCGA'):
                            return 'TCGA'
                        elif x.endswith('_GTEX'):
                            return 'GTEX'
                        return 'UNKNOWN'


                    for df_name, df in [('train', train_data), ('val', val_data), ('test', test_data)]:
                        if 'SOURCE' not in df.columns:
                            df['SOURCE'] = df['LABEL'].apply(get_source_from_label)

                    print("✓ All checkpoints loaded. Proceeding directly to model training...")
                    skip_to_training = True
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            print("Deleting corrupted checkpoint and regenerating...")
            if os.path.exists(merged_data_scaled_file):
                os.remove(merged_data_scaled_file)
            skip_to_training = False
    elif os.path.exists(merged_data_preprocessed_file):
        # Preprocessed checkpoint exists - load it and continue with scaling/SMOTE
        with open(merged_data_preprocessed_file, 'rb') as f:
            pre_data = pickle.load(f)
        merged_data = pre_data['merged_data']
        common_genes = pre_data['common_genes']  # Set common_genes for later use
        feature_columns = pre_data['common_genes']
        train_data = pre_data['train_data']
        val_data = pre_data['val_data']
        test_data = pre_data['test_data']
        label_encoder = pre_data['label_encoder']


        # Ensure SOURCE column exists
        def get_source_from_label(x):
            if x.endswith('_METASTATIC'):
                return 'METASTATIC'
            elif x.endswith('_TCGA'):
                return 'TCGA'
            elif x.endswith('_GTEX'):
                return 'GTEX'
            return 'UNKNOWN'


        for df_name, df in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if 'SOURCE' not in df.columns:
                df['SOURCE'] = df['LABEL'].apply(get_source_from_label)

        print("Preprocessed data loaded. Continuing with scaling and SMOTE...")
        skip_to_training = False
    else:
        # No checkpoint exists - do full preprocessing
        skip_to_training = False
        print("No checkpoint found. Starting full data loading and preprocessing...")
        print("Preprocessed data not found. Starting data loading and preprocessing...")
        with timer("Data Loading"):
            patient_data = pd.read_csv(patient_data_file)
            metastatic_data = pd.read_csv(metastatic_data_file)

            # Process GTEx with donor-aware approach
            gtex_data_full = process_gtex_files_donor_aware(gtex_reads_dir, gtex_processed_file)

        # Convert columns to uppercase
        patient_data.columns = patient_data.columns.str.upper()
        gtex_data_full.columns = gtex_data_full.columns.str.upper()
        metastatic_data.columns = metastatic_data.columns.str.upper()

        # ============================================================================
        # PREPROCESSING: Full preprocessing (no checkpoint exists)
        # ============================================================================
        # ============================================================================
        # STEP 1: Remove duplicate genes (columns) and duplicate samples (rows)
        # ============================================================================
        print("\n" + "=" * 70)
        print("STEP 1: Removing duplicate genes and samples from all datasets")
        print("=" * 70)

        # --- TCGA (Patient) Data ---
        print("\n[TCGA] Checking for duplicates...")
        initial_rows_tcga = len(patient_data)
        initial_cols_tcga = len(patient_data.columns)

        if patient_data.columns.duplicated().any():
            dup_cols = patient_data.columns[patient_data.columns.duplicated()].tolist()
            print(
                f"  Found {len(dup_cols)} duplicate gene columns: {dup_cols[:10]}{'...' if len(dup_cols) > 10 else ''}")
            patient_data = patient_data.loc[:, ~patient_data.columns.duplicated()]
            print(f"  Removed duplicate columns. Columns: {initial_cols_tcga} -> {len(patient_data.columns)}")

        dup_rows_tcga = patient_data.duplicated()
        if dup_rows_tcga.any():
            num_dup = dup_rows_tcga.sum()
            print(f"  Found {num_dup} duplicate samples (exact row matches)")
            patient_data = patient_data[~dup_rows_tcga]
            print(f"  Removed duplicate samples. Rows: {initial_rows_tcga} -> {len(patient_data)}")

        if not patient_data.columns.duplicated().any() and not patient_data.duplicated().any():
            print(f"  ✓ No duplicates found. Final: {len(patient_data)} samples, {len(patient_data.columns)} columns")

        # --- GTEx Data ---
        print("\n[GTEx] Checking for duplicates...")
        initial_rows_gtex = len(gtex_data_full)
        initial_cols_gtex = len(gtex_data_full.columns)

        if gtex_data_full.columns.duplicated().any():
            dup_cols = gtex_data_full.columns[gtex_data_full.columns.duplicated()].tolist()
            print(
                f"  Found {len(dup_cols)} duplicate gene columns: {dup_cols[:10]}{'...' if len(dup_cols) > 10 else ''}")
            gtex_data_full = gtex_data_full.loc[:, ~gtex_data_full.columns.duplicated()]
            print(f"  Removed duplicate columns. Columns: {initial_cols_gtex} -> {len(gtex_data_full.columns)}")

        dup_rows_gtex = gtex_data_full.duplicated()
        if dup_rows_gtex.any():
            num_dup = dup_rows_gtex.sum()
            print(f"  Found {num_dup} duplicate samples (exact row matches)")
            gtex_data_full = gtex_data_full[~dup_rows_gtex]
            print(f"  Removed duplicate samples. Rows: {initial_rows_gtex} -> {len(gtex_data_full)}")

        if not gtex_data_full.columns.duplicated().any() and not gtex_data_full.duplicated().any():
            print(
                f"  ✓ No duplicates found. Final: {len(gtex_data_full)} samples, {len(gtex_data_full.columns)} columns")

        # --- Metastatic Data ---
        print("\n[Metastatic] Checking for duplicates...")
        initial_rows_met = len(metastatic_data)
        initial_cols_met = len(metastatic_data.columns)

        if metastatic_data.columns.duplicated().any():
            dup_cols = metastatic_data.columns[metastatic_data.columns.duplicated()].tolist()
            print(
                f"  Found {len(dup_cols)} duplicate gene columns: {dup_cols[:10]}{'...' if len(dup_cols) > 10 else ''}")
            metastatic_data = metastatic_data.loc[:, ~metastatic_data.columns.duplicated()]
            print(f"  Removed duplicate columns. Columns: {initial_cols_met} -> {len(metastatic_data.columns)}")

        dup_rows_met = metastatic_data.duplicated()
        if dup_rows_met.any():
            num_dup = dup_rows_met.sum()
            print(f"  Found {num_dup} duplicate samples (exact row matches)")
            metastatic_data = metastatic_data[~dup_rows_met]
            print(f"  Removed duplicate samples. Rows: {initial_rows_met} -> {len(metastatic_data)}")

        if not metastatic_data.columns.duplicated().any() and not metastatic_data.duplicated().any():
            print(
                f"  ✓ No duplicates found. Final: {len(metastatic_data)} samples, {len(metastatic_data.columns)} columns")

        print("\n" + "=" * 70)
        print("Duplicate removal complete. Proceeding to gene identification...")
        print("=" * 70 + "\n")

        # Identify gene columns (after duplicate removal)
        non_gene_columns_patient = ['CANCER TYPE', 'CLASS']
        gene_columns_patient = [col for col in patient_data.columns if col not in non_gene_columns_patient]

        # GTEx: exclude metadata columns (DONOR_ID, LABEL, SAMPLE_ID)
        non_gene_columns_gtex = ['DONOR_ID', 'LABEL', 'SAMPLE_ID']
        gene_columns_gtex = [col for col in gtex_data_full.columns if col not in non_gene_columns_gtex]

        non_gene_columns_metastatic = ['CANCERTYPE']
        gene_columns_metastatic = [col for col in metastatic_data.columns if col not in non_gene_columns_metastatic]

        # ============================================================================
        # STEP 2: Find common genes and filter datasets BEFORE imputation
        # ============================================================================
        print("\n" + "=" * 70)
        print("STEP 2: Finding common genes and filtering datasets (BEFORE imputation)")
        print("=" * 70)
        with timer("STEP 2: Common Gene Filtering"):
            print(f"\nGene counts before filtering:")
            print(f"  TCGA: {len(gene_columns_patient):,} genes")
            print(f"  GTEx: {len(gene_columns_gtex):,} genes")
            print(f"  Metastatic: {len(gene_columns_metastatic):,} genes")

            # Find common genes (intersection of all three datasets)
            common_genes = sorted(list(
                set(gene_columns_patient)
                & set(gene_columns_gtex)
                & set(gene_columns_metastatic)
            ))
        print(f"\nCommon genes across all datasets: {len(common_genes):,}")

        if len(common_genes) == 0:
            raise ValueError("ERROR: No common genes found across datasets! Check data files.")

        # Filter each dataset to only common genes + metadata columns (BEFORE imputation)
        print("\nFiltering datasets to common genes only...")

        # TCGA: keep common genes + metadata
        patient_metadata_cols = [col for col in patient_data.columns if col in non_gene_columns_patient]
        patient_data_filtered = patient_data[common_genes + patient_metadata_cols].copy()
        print(
            f"  TCGA: {len(patient_data)} samples × {len(patient_data.columns)} cols -> {len(patient_data_filtered)} samples × {len(patient_data_filtered.columns)} cols")

        # GTEx: keep common genes + metadata
        gtex_metadata_cols = [col for col in gtex_data_full.columns if col in non_gene_columns_gtex]
        gtex_data_filtered = gtex_data_full[common_genes + gtex_metadata_cols].copy()
        print(
            f"  GTEx: {len(gtex_data_full)} samples × {len(gtex_data_full.columns)} cols -> {len(gtex_data_filtered)} samples × {len(gtex_data_filtered.columns)} cols")

        # Metastatic: keep common genes + metadata
        metastatic_metadata_cols = [col for col in metastatic_data.columns if col in non_gene_columns_metastatic]
        metastatic_data_filtered = metastatic_data[common_genes + metastatic_metadata_cols].copy()
        print(
            f"  Metastatic: {len(metastatic_data)} samples × {len(metastatic_data.columns)} cols -> {len(metastatic_data_filtered)} samples × {len(metastatic_data_filtered.columns)} cols")

        print("\n" + "=" * 70)
        print("Datasets filtered to common genes. Proceeding to imputation...")
        print("=" * 70 + "\n")

        # Update variable names to use filtered datasets
        patient_data = patient_data_filtered
        gtex_data_full = gtex_data_filtered
        metastatic_data = metastatic_data_filtered

        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=5)

        # ============================================================================
        # STEP 3: Impute missing values (on already-filtered common genes only)
        # ============================================================================
        print("STEP 3: Imputing missing values (KNN imputation on common genes only)...")
        with timer("STEP 3: KNN Imputation"):
            # -------------------
            # 1) Impute TCGA (patient)
            # -------------------
            # Datasets are already filtered to common_genes, so just select gene columns
            patient_expr = patient_data[common_genes]
            patient_expr_imputed = imputer.fit_transform(patient_expr)
            common_genes_ordered = list(patient_expr.columns)
        patient_expr_imputed = pd.DataFrame(patient_expr_imputed, columns=common_genes_ordered,
                                            index=patient_expr.index)
        print(f"  TCGA imputation complete: {len(patient_expr_imputed)} samples × {len(common_genes_ordered)} genes")

        # -------------------
        # 2) Impute GTEX
        # -------------------
        gtex_expr = gtex_data_full.loc[:, common_genes_ordered]
        gtex_expr_imputed = imputer.transform(gtex_expr)
        gtex_expr_imputed = pd.DataFrame(gtex_expr_imputed, columns=common_genes_ordered, index=gtex_data_full.index)
        print(f"  GTEx imputation complete: {len(gtex_expr_imputed)} samples × {len(common_genes_ordered)} genes")

        # -------------------
        # 3) Impute Metastatic
        # -------------------
        metastatic_expr = metastatic_data.loc[:, common_genes_ordered]
        metastatic_expr_imputed = imputer.transform(metastatic_expr)
        metastatic_expr_imputed = pd.DataFrame(metastatic_expr_imputed, columns=common_genes_ordered,
                                               index=metastatic_expr.index)
        print(
            f"  Metastatic imputation complete: {len(metastatic_expr_imputed)} samples × {len(common_genes_ordered)} genes")

        print("Imputation complete for all datasets.\n")

        # Assign SOURCE, LABEL, and TISSUE columns
        patient_labels = patient_data['CANCER TYPE'].reset_index(drop=True)
        gtex_labels = gtex_data_full['LABEL'].reset_index(drop=True)  # Already in format TissueName_GTEx
        metastatic_labels = metastatic_data['CANCERTYPE'].reset_index(drop=True)

        patient_expr_imputed.index = patient_labels.index
        gtex_expr_imputed.index = gtex_labels.index
        metastatic_expr_imputed.index = metastatic_labels.index

        patient_expr_imputed['SOURCE'] = 'TCGA'
        gtex_expr_imputed['SOURCE'] = 'GTEX'
        metastatic_expr_imputed['SOURCE'] = 'METASTATIC'

        patient_expr_imputed['LABEL'] = patient_labels.values + '_TCGA'
        gtex_expr_imputed['LABEL'] = gtex_labels.values  # Already formatted as TissueName_GTEx
        metastatic_expr_imputed['LABEL'] = metastatic_labels.values + '_METASTATIC'
        metastatic_expr_imputed['TISSUE'] = metastatic_labels.values

        # Add SAMPLE_ID and DONOR_ID to GTEx (for splitting and leakage checking)
        gtex_expr_imputed['SAMPLE_ID'] = gtex_data_full['SAMPLE_ID'].values
        gtex_expr_imputed['DONOR_ID'] = gtex_data_full['DONOR_ID'].values

        # -------------------
        # 4) Apply "≥30" filter only to TCGA+GTEX (temporarily merge to find common labels)
        # -------------------
        patient_gtex_merged = pd.concat([patient_expr_imputed, gtex_expr_imputed], ignore_index=True)
        pg_label_counts = patient_gtex_merged['LABEL'].value_counts()
        pg_sufficient_labels = pg_label_counts[pg_label_counts >= 30].index.tolist()

        # Keep only those "≥30" labels in patient+gtex
        patient_gtex_merged = patient_gtex_merged[patient_gtex_merged['LABEL'].isin(pg_sufficient_labels)].reset_index(
            drop=True)

        # Re-split back into patient/gtex
        patient_expr_imputed = patient_gtex_merged[patient_gtex_merged['SOURCE'] == 'TCGA'].copy().reset_index(
            drop=True)
        gtex_expr_imputed = patient_gtex_merged[patient_gtex_merged['SOURCE'] == 'GTEX'].copy().reset_index(drop=True)

        print(f"After ≥30 filter: {len(patient_expr_imputed):,} TCGA samples, {len(gtex_expr_imputed):,} GTEx samples")

        # -------------------
        # 5) Filter out metastatic labels that have <5 samples
        # -------------------
        meta_label_counts = metastatic_expr_imputed['LABEL'].value_counts()
        # keep only those MET labels with >=5
        keep_met_labels = meta_label_counts[meta_label_counts >= 5].index.tolist()
        metastatic_expr_imputed = metastatic_expr_imputed[
            metastatic_expr_imputed['LABEL'].isin(keep_met_labels)].reset_index(drop=True)

        # -------------------
        # 6) Held-out test set from 4 metastatic classes
        # -------------------
        target_test_classes = ['LUAD', 'BRCA', 'PAAD', 'COAD']


        def safe_sample(lst, n):
            if len(lst) <= n:
                return lst
            return random.sample(lst, n)


        luad_inds = metastatic_expr_imputed[metastatic_expr_imputed['TISSUE'] == 'LUAD'].index.tolist()
        paad_inds = metastatic_expr_imputed[metastatic_expr_imputed['TISSUE'] == 'PAAD'].index.tolist()
        brca_inds = metastatic_expr_imputed[metastatic_expr_imputed['TISSUE'] == 'BRCA'].index.tolist()
        coad_inds = metastatic_expr_imputed[metastatic_expr_imputed['TISSUE'] == 'COAD'].index.tolist()

        luad_test = safe_sample(luad_inds, 20)
        paad_test = safe_sample(paad_inds, 25)
        brca_test = safe_sample(brca_inds, 40)
        coad_test = safe_sample(coad_inds, 35)

        test_samples = pd.concat([
            metastatic_expr_imputed.loc[luad_test],
            metastatic_expr_imputed.loc[paad_test],
            metastatic_expr_imputed.loc[brca_test],
            metastatic_expr_imputed.loc[coad_test]
        ], ignore_index=True)

        # Remove them from leftover metastatic
        metastatic_expr_imputed = metastatic_expr_imputed.drop(test_samples.index)

        # -------------------
        # 7) 75:25 split of leftover metastatic
        #    (including leftover of those 4 classes + other metastatic)
        # -------------------
        if len(metastatic_expr_imputed) > 0:
            train_meta, val_meta = train_test_split(
                metastatic_expr_imputed,
                test_size=0.25,
                random_state=SEED,
                stratify=metastatic_expr_imputed['LABEL']
            )
        else:
            train_meta = pd.DataFrame(columns=metastatic_expr_imputed.columns)
            val_meta = pd.DataFrame(columns=metastatic_expr_imputed.columns)

        # -------------------
        # 8) Split GTEx by DONOR ID (60:20:20) - NO LEAKAGE
        # -------------------
        print("\nSplitting GTEx data by donor ID (60:20:20) to prevent leakage...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=SEED)
        gtex_train_idx, gtex_temp_idx = next(
            gss.split(gtex_expr_imputed, gtex_expr_imputed['LABEL'], groups=gtex_expr_imputed['DONOR_ID']))

        # Split temp into val and test (50/50 of the 40% = 20% each)
        gtex_temp_df = gtex_expr_imputed.iloc[gtex_temp_idx]
        gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
        gtex_val_idx_temp, gtex_test_idx_temp = next(
            gss_temp.split(gtex_temp_df, gtex_temp_df['LABEL'], groups=gtex_temp_df['DONOR_ID']))

        gtex_train = gtex_expr_imputed.iloc[gtex_train_idx].copy()
        gtex_val = gtex_temp_df.iloc[gtex_val_idx_temp].copy()
        gtex_test = gtex_temp_df.iloc[gtex_test_idx_temp].copy()

        print(f"GTEx splits: Train={len(gtex_train):,} ({len(gtex_train) / len(gtex_expr_imputed) * 100:.1f}%), "
              f"Val={len(gtex_val):,} ({len(gtex_val) / len(gtex_expr_imputed) * 100:.1f}%), "
              f"Test={len(gtex_test):,} ({len(gtex_test) / len(gtex_expr_imputed) * 100:.1f}%)")

        # -------------------
        # 9) Split Patient (TCGA) by label stratified (60:20:20) - EXACTLY AS CURRENT
        # -------------------
        print("\nSplitting TCGA data by label stratified (60:20:20)...")
        X_patient = patient_expr_imputed
        y_patient = patient_expr_imputed['LABEL']

        patient_train, patient_temp = train_test_split(
            X_patient, test_size=0.4, random_state=SEED, stratify=y_patient
        )
        y_patient_temp = patient_temp['LABEL']
        patient_val, patient_test = train_test_split(
            patient_temp, test_size=0.5, random_state=SEED, stratify=y_patient_temp
        )

        print(
            f"TCGA splits: Train={len(patient_train):,} ({len(patient_train) / len(patient_expr_imputed) * 100:.1f}%), "
            f"Val={len(patient_val):,} ({len(patient_val) / len(patient_expr_imputed) * 100:.1f}%), "
            f"Test={len(patient_test):,} ({len(patient_test) / len(patient_expr_imputed) * 100:.1f}%)")

        # -------------------
        # 10) Merge final splits from all three datasets
        # -------------------
        print("\nMerging final splits from GTEx, TCGA, and Metastatic datasets...")

        train_data = pd.concat([gtex_train, patient_train, train_meta], ignore_index=True)
        val_data = pd.concat([gtex_val, patient_val, val_meta], ignore_index=True)
        test_data = pd.concat([gtex_test, patient_test, test_samples], ignore_index=True)

        print(f"Final merged splits: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")

        # Ensure unique SAMPLE_ID AFTER merging (like original code)
        if 'SAMPLE_ID' not in train_data.columns or train_data['SAMPLE_ID'].isna().any():
            # Create SAMPLE_IDs for all samples, preserving existing ones where they exist
            train_data['SAMPLE_ID'] = [f"TRAIN_{i}" for i in range(len(train_data))]
        if 'SAMPLE_ID' not in val_data.columns or val_data['SAMPLE_ID'].isna().any():
            val_data['SAMPLE_ID'] = [f"VAL_{i}" for i in range(len(val_data))]
        if 'SAMPLE_ID' not in test_data.columns or test_data['SAMPLE_ID'].isna().any():
            test_data['SAMPLE_ID'] = [f"TEST_{i}" for i in range(len(test_data))]

        # -------------------
        # 11) Unit Test: Check for data leakage (SAMPLE_ID and DONOR_ID)
        # -------------------
        print("\n" + "=" * 80)
        print("UNIT TEST: Checking for data leakage...")
        print("=" * 80)

        train_sample_ids = set(train_data['SAMPLE_ID'].dropna().astype(str))
        val_sample_ids = set(val_data['SAMPLE_ID'].dropna().astype(str))
        test_sample_ids = set(test_data['SAMPLE_ID'].dropna().astype(str))

        train_val_overlap = train_sample_ids & val_sample_ids
        train_test_overlap = train_sample_ids & test_sample_ids
        val_test_overlap = val_sample_ids & test_sample_ids

        if train_val_overlap:
            print(f"✗ ERROR: {len(train_val_overlap)} SAMPLE_IDs overlap between train and val!")
            print(f"  Examples: {list(train_val_overlap)[:5]}")
            raise ValueError("Data leakage detected: SAMPLE_ID overlap between train and val")
        if train_test_overlap:
            print(f"✗ ERROR: {len(train_test_overlap)} SAMPLE_IDs overlap between train and test!")
            print(f"  Examples: {list(train_test_overlap)[:5]}")
            raise ValueError("Data leakage detected: SAMPLE_ID overlap between train and test")
        if val_test_overlap:
            print(f"✗ ERROR: {len(val_test_overlap)} SAMPLE_IDs overlap between val and test!")
            print(f"  Examples: {list(val_test_overlap)[:5]}")
            raise ValueError("Data leakage detected: SAMPLE_ID overlap between val and test")

        print("✓ No SAMPLE_ID leakage detected")

        # Check DONOR_ID leakage (for GTEx samples)
        if 'DONOR_ID' in train_data.columns:
            train_donor_ids = set(train_data[train_data['SOURCE'] == 'GTEX']['DONOR_ID'].dropna().astype(str))
        else:
            train_donor_ids = set()

        if 'DONOR_ID' in val_data.columns:
            val_donor_ids = set(val_data[val_data['SOURCE'] == 'GTEX']['DONOR_ID'].dropna().astype(str))
        else:
            val_donor_ids = set()

        if 'DONOR_ID' in test_data.columns:
            test_donor_ids = set(test_data[test_data['SOURCE'] == 'GTEX']['DONOR_ID'].dropna().astype(str))
        else:
            test_donor_ids = set()

        train_val_donor_overlap = train_donor_ids & val_donor_ids
        train_test_donor_overlap = train_donor_ids & test_donor_ids
        val_test_donor_overlap = val_donor_ids & test_donor_ids

        if train_val_donor_overlap:
            print(f"✗ ERROR: {len(train_val_donor_overlap)} DONOR_IDs overlap between train and val!")
            print(f"  Examples: {list(train_val_donor_overlap)[:5]}")
            raise ValueError("Data leakage detected: DONOR_ID overlap between train and val")
        if train_test_donor_overlap:
            print(f"✗ ERROR: {len(train_test_donor_overlap)} DONOR_IDs overlap between train and test!")
            print(f"  Examples: {list(train_test_donor_overlap)[:5]}")
            raise ValueError("Data leakage detected: DONOR_ID overlap between train and test")
        if val_test_donor_overlap:
            print(f"✗ ERROR: {len(val_test_donor_overlap)} DONOR_IDs overlap between val and test!")
            print(f"  Examples: {list(val_test_donor_overlap)[:5]}")
            raise ValueError("Data leakage detected: DONOR_ID overlap between val and test")

        print("✓ No DONOR_ID leakage detected")
        print("=" * 80)

        # Remove duplicates across splits by SAMPLE_ID (safety check)
        train_ids = set(train_data['SAMPLE_ID'].dropna().astype(str))
        if len(train_ids) == 0:
            print("WARNING: No valid SAMPLE_IDs in training data after NaN removal")
        else:
            val_data = val_data[~val_data['SAMPLE_ID'].isin(train_ids)]
            val_ids = set(val_data['SAMPLE_ID'].dropna().astype(str))
            if len(val_ids) == 0:
                print("WARNING: No valid SAMPLE_IDs in validation data after filtering")
        test_data = test_data[~test_data['SAMPLE_ID'].isin(train_ids)]
        test_data = test_data[~test_data['SAMPLE_ID'].isin(val_ids)]
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # -------------------
        # 12) Save individual dataset files with split information
        # -------------------
        # This section only runs during preprocessing (inside the else block)
        # When loading from checkpoint, these files should already exist
        # Check if individual files already exist - if so, skip saving
        gtex_save_file = os.path.join(new_output_dir, "GTEx.pkl")
        tcga_save_file = os.path.join(new_output_dir, "TCGA.pkl")
        metastatic_save_file = os.path.join(new_output_dir, "metastatic.pkl")

        if os.path.exists(gtex_save_file) and os.path.exists(tcga_save_file) and os.path.exists(metastatic_save_file):
            print("\nIndividual dataset files (GTEx.pkl, TCGA.pkl, metastatic.pkl) already exist. Skipping save step.")
        else:
            # Files don't exist, so we need to create them (only possible during preprocessing)
            # Variables should exist since we're in the preprocessing else block
            print("\nSaving individual dataset files with split information...")

        # Add split column to each dataset's splits
        gtex_train_with_split = gtex_train.copy()
        gtex_val_with_split = gtex_val.copy()
        gtex_test_with_split = gtex_test.copy()
        gtex_train_with_split['SPLIT'] = 'train'
        gtex_val_with_split['SPLIT'] = 'val'
        gtex_test_with_split['SPLIT'] = 'test'
        gtex_all_splits = pd.concat([gtex_train_with_split, gtex_val_with_split, gtex_test_with_split],
                                    ignore_index=True)
        if not os.path.exists(gtex_save_file):
            with open(gtex_save_file, 'wb') as f:
                pickle.dump(gtex_all_splits, f)
            print(f"  GTEx data with splits saved to: {gtex_save_file}")

        patient_train_with_split = patient_train.copy()
        patient_val_with_split = patient_val.copy()
        patient_test_with_split = patient_test.copy()
        patient_train_with_split['SPLIT'] = 'train'
        patient_val_with_split['SPLIT'] = 'val'
        patient_test_with_split['SPLIT'] = 'test'
        patient_all_splits = pd.concat([patient_train_with_split, patient_val_with_split, patient_test_with_split],
                                       ignore_index=True)
        if not os.path.exists(tcga_save_file):
            with open(tcga_save_file, 'wb') as f:
                pickle.dump(patient_all_splits, f)
            print(f"  TCGA data with splits saved to: {tcga_save_file}")

        train_meta_with_split = train_meta.copy()
        val_meta_with_split = val_meta.copy()
        test_samples_with_split = test_samples.copy()
        train_meta_with_split['SPLIT'] = 'train'
        val_meta_with_split['SPLIT'] = 'val'
        test_samples_with_split['SPLIT'] = 'test'
        metastatic_all_splits = pd.concat([train_meta_with_split, val_meta_with_split, test_samples_with_split],
                                          ignore_index=True)
        if not os.path.exists(metastatic_save_file):
            with open(metastatic_save_file, 'wb') as f:
                pickle.dump(metastatic_all_splits, f)
            print(f"  Metastatic data with splits saved to: {metastatic_save_file}")

        # Remove DONOR_ID from final merged dataframes (only needed for splitting, not for training)
        if 'DONOR_ID' in train_data.columns:
            train_data = train_data.drop(columns=['DONOR_ID'])
        if 'DONOR_ID' in val_data.columns:
            val_data = val_data.drop(columns=['DONOR_ID'])
        if 'DONOR_ID' in test_data.columns:
            test_data = test_data.drop(columns=['DONOR_ID'])
        print("  DONOR_ID removed from final merged dataframes (kept in individual GTEx.pkl file)")

        # 13) Label-encode only what's left
        train_labels_set = set(train_data['LABEL'].unique())
        val_data = val_data[val_data['LABEL'].isin(train_labels_set)].reset_index(drop=True)
        test_data = test_data[test_data['LABEL'].isin(train_labels_set)].reset_index(drop=True)

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        all_labels_uniq = pd.concat([
            train_data['LABEL'],
            val_data['LABEL'],
            test_data['LABEL']
        ]).unique()
        label_encoder.fit(all_labels_uniq)
        train_data['LABEL_NUMERIC'] = label_encoder.transform(train_data['LABEL'])
        val_data['LABEL_NUMERIC'] = label_encoder.transform(val_data['LABEL'])
        test_data['LABEL_NUMERIC'] = label_encoder.transform(test_data['LABEL'])

        # 14) log2 transform on final sets (vectorized for speed)
        print("Applying log2 transform to gene expression data...")
        train_data[common_genes] = np.log2(train_data[common_genes] + 1)
        val_data[common_genes] = np.log2(val_data[common_genes] + 1)
        test_data[common_genes] = np.log2(test_data[common_genes] + 1)
        print("Log2 transform complete.")

        label_counts_after_smote = train_data['LABEL'].value_counts().to_dict()


        # Re-assign SOURCE & LABEL_NUMERIC after SMOTE
        def get_source_from_label(x):
            if x.endswith('_METASTATIC'):
                return 'METASTATIC'
            elif x.endswith('_TCGA'):
                return 'TCGA'
            elif x.endswith('_GTEX'):
                return 'GTEX'
            return 'UNKNOWN'


        train_data['SOURCE'] = train_data['LABEL'].apply(get_source_from_label)
        # Re-encode numeric
        train_data['LABEL_NUMERIC'] = label_encoder.transform(train_data['LABEL'])

        # 13) Generate "split_label_counts.csv"
        train_counts = train_data['LABEL'].value_counts().to_dict()
        val_counts = val_data['LABEL'].value_counts().to_dict()
        test_counts = test_data['LABEL'].value_counts().to_dict()

        metastatic_labels_for_smote = [
            lbl for lbl in train_data['LABEL'].unique()
            if lbl.endswith('_METASTATIC') and train_counts.get(lbl, 0) >= 5
        ]

        rows_list = []
        all_labels_final = set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())
        for lbl in sorted(all_labels_final):
            row_ = {
                'Label': lbl,
                'TrainCount': train_counts.get(lbl, 0),
                'ValCount': val_counts.get(lbl, 0),
                'TestCount': test_counts.get(lbl, 0)
            }
            if lbl in metastatic_labels_for_smote:
                row_['SMOTE'] = True
                row_['SMOTE_Generated'] = 'Applied after scaling'  # Will be updated after SMOTE
            else:
                row_['SMOTE'] = False
                row_['SMOTE_Generated'] = 0
            rows_list.append(row_)

        df_split_counts = pd.DataFrame(rows_list)
        split_counts_csv = os.path.join(output_base_dir, 'split_label_counts.csv')
        df_split_counts.to_csv(split_counts_csv, index=False)
        print(f"split_label_counts CSV saved => {split_counts_csv}")

        # Merge final data for reference
        merged_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    
        with open(merged_data_preprocessed_file, 'wb') as f:
            pickle.dump({
                'merged_data': merged_data,
                'common_genes': common_genes_ordered,  # Use ordered version for consistency
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'label_encoder': label_encoder
            }, f)
        print(
            "Preprocessed data saved. Metastatic labels <5 removed, 4-class test extracted, leftover 75:25 splitted, SMOTE applied, CSV stored.")

        pre_data = {
            'merged_data': merged_data,
            'common_genes': common_genes_ordered,  # Use ordered version for consistency
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'label_encoder': label_encoder
        }

# Only do feature selection and scaling if we didn't already load from final checkpoint
if not skip_to_training:
    # After preprocessing (or loading from checkpoint), ensure SOURCE column exists
    if 'SOURCE' not in train_data.columns:
        def get_source_from_label(x):
            if x.endswith('_METASTATIC'):
                return 'METASTATIC'
            elif x.endswith('_TCGA'):
                return 'TCGA'
            elif x.endswith('_GTEX'):
                return 'GTEX'
            return 'UNKNOWN'


        train_data['SOURCE'] = train_data['LABEL'].apply(get_source_from_label)
        val_data['SOURCE'] = val_data['LABEL'].apply(get_source_from_label)
        test_data['SOURCE'] = test_data['LABEL'].apply(get_source_from_label)

    # Remove DONOR_ID if it exists (should already be removed, but safety check)
    if 'DONOR_ID' in train_data.columns:
        train_data = train_data.drop(columns=['DONOR_ID'])
    if 'DONOR_ID' in val_data.columns:
        val_data = val_data.drop(columns=['DONOR_ID'])
    if 'DONOR_ID' in test_data.columns:
        test_data = test_data.drop(columns=['DONOR_ID'])


    print("Performing feature selection with Lasso (BEFORE scaling and SMOTE)...")
    print("IMPORTANT: LASSO feature selection is performed EXCLUSIVELY on real training data")
    print("          (no SMOTE samples, no test/val data) to prevent data leakage.")

    max_genes = 600  # <-- ADJUST THIS TO YOUR DESIRED MAX NUMBER OF GENES

    if os.path.exists(feature_sets_file):
        with open(feature_sets_file, 'rb') as f:
            feature_sets = pickle.load(f)
            print(f"Feature sets loaded from checkpoint: {len(feature_sets['FullLasso'])} features")
    else:
        with timer("Feature Selection (Lasso)"):
            # ========================================================================
            # Feature selection uses ONLY real training data (no SMOTE, no test/val)
            # ========================================================================
            train_data_for_fs = pre_data['train_data'].copy()  # Only real training data for feature selection
            genes_for_selection = pre_data['common_genes']

            X_fs = train_data_for_fs[genes_for_selection].values  # Real training features only
            y_fs = train_data_for_fs['LABEL_NUMERIC'].values  # Real training labels only

            # Note: pre_data['val_data'] and pre_data['test_data'] are NOT used here
            # This ensures no information from test/validation sets leaks into feature selection

            # Lasso without cross-validation (fitted on real training data only)
            lasso_model = Lasso(alpha=0.001, random_state=SEED, max_iter=10000)

            # Use SelectFromModel to ensure no more than 'max_genes' are selected
            sfm = SelectFromModel(estimator=lasso_model,
                                  max_features=max_genes,
                                  threshold=-np.inf)  # -np.inf => select up to max_features

            # Fit LASSO on REAL TRAINING DATA ONLY (X_fs, y_fs are from real train_data_local)
            sfm.fit(X_fs, y_fs)
            selected_mask = sfm.get_support()
            selected_genes = [g for g, use_g in zip(genes_for_selection, selected_mask) if use_g]

            # If no genes got selected, fall back to all genes (rare edge case)
            if len(selected_genes) == 0:
                selected_genes = genes_for_selection

            feature_sets = {'FullLasso': selected_genes}
            with open(feature_sets_file, 'wb') as f:
                pickle.dump(feature_sets, f)
        print(f"Feature selection complete: {len(selected_genes)} features selected from {len(genes_for_selection)}")

    feature_columns = feature_sets['FullLasso']
    print("Number of features selected:", len(feature_columns))

    # ============================================================================
    # STEP 2: Scaling (AFTER feature selection, BEFORE SMOTE)
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Standard Scaling (after feature selection, before SMOTE)")
    print("=" * 80)

    print("Checking or creating scaled data with SMOTE if needed...")
    if os.path.exists(merged_data_scaled_file):
        with open(merged_data_scaled_file, 'rb') as f:
            data_scaled = pickle.load(f)
        train_data_scaled = data_scaled['train_data_scaled']
        val_data_scaled = data_scaled['val_data_scaled']
        test_data_scaled = data_scaled['test_data_scaled']
        scaler = data_scaled['scaler']
        print("Scaled data loaded from checkpoint.")
    else:
        with timer("Standard Scaling"):
            train_data_local = pre_data['train_data'].copy()
            val_data_local = pre_data['val_data'].copy()
            test_data_local = pre_data['test_data'].copy()

            # Fit scaler on REAL training data only (no SMOTE samples yet)
            scaler = StandardScaler()
            scaler.fit(train_data_local[feature_columns])
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)

            train_data_scaled = train_data_local.copy()
            val_data_scaled = val_data_local.copy()
            test_data_scaled = test_data_local.copy()

            if 'SAMPLE_ID' not in train_data_scaled.columns:
                if 'SAMPLE_ID' in train_data_local.columns:
                    train_data_scaled['SAMPLE_ID'] = train_data_local['SAMPLE_ID']
                else:
                    train_data_scaled['SAMPLE_ID'] = [f"TRAINscaled_{i}" for i in range(len(train_data_scaled))]

            if 'SAMPLE_ID' not in val_data_scaled.columns:
                if 'SAMPLE_ID' in val_data_local.columns:
                    val_data_scaled['SAMPLE_ID'] = val_data_local['SAMPLE_ID']
                else:
                    val_data_scaled['SAMPLE_ID'] = [f"VALscaled_{i}" for i in range(len(val_data_scaled))]

            if 'SAMPLE_ID' not in test_data_scaled.columns:
                if 'SAMPLE_ID' in test_data_local.columns:
                    test_data_scaled['SAMPLE_ID'] = test_data_local['SAMPLE_ID']
                else:
                    test_data_scaled['SAMPLE_ID'] = [f"TESTscaled_{i}" for i in range(len(test_data_scaled))]

            # Now  standard scaling on the selected feature columns
            train_data_scaled[feature_columns] = scaler.transform(train_data_local[feature_columns])
            val_data_scaled[feature_columns] = scaler.transform(val_data_local[feature_columns])
            test_data_scaled[feature_columns] = scaler.transform(test_data_local[feature_columns])

            # ============================================================================
            # STEP 3: SMOTE on metastatic classes in train (AFTER scaling, AFTER feature selection)
            # ============================================================================
            print("\n" + "=" * 80)
            print("STEP 3: SMOTE on metastatic classes (after scaling and feature selection)")
            print("=" * 80)
            with timer("SMOTE Oversampling"):
                train_metas = train_data_scaled[train_data_scaled['LABEL'].str.contains('_METASTATIC')]
                class_counts = train_metas['LABEL'].value_counts()
                classes_for_smote = class_counts[class_counts >= 5].index.tolist()
                if len(classes_for_smote) > 0:
                    # FIXED: Use adaptive k_neighbors based on each class size
                    # For each class, use min(5, class_size-1) neighbors, but ensure at least 1
                    smote_strategy = {}
                    k_neighbors_list = []
                    for cls in classes_for_smote:
                        class_size = class_counts[cls]
                        smote_strategy[cls] = int(np.ceil(class_size * 1.5))
                        # Use all available neighbors when class size is small, but cap at 5
                        k_neighbors_list.append(min(5, max(1, class_size - 1)))

                    # Use the minimum k_neighbors to ensure SMOTE works for all classes
                    k_neighbors_smote = min(k_neighbors_list) if k_neighbors_list else 1

                    print(f"  Applying SMOTE to {len(classes_for_smote)} metastatic classes")
                    print(f"  Using k_neighbors={k_neighbors_smote} (adaptive based on class sizes)")

                    sm = SMOTE(sampling_strategy=smote_strategy,
                               random_state=SEED,
                               k_neighbors=k_neighbors_smote)
                    X_sm = train_data_scaled[feature_columns]
                    y_sm = train_data_scaled['LABEL']
                    mask_for_smote = train_data_scaled['LABEL'].isin(classes_for_smote)
                    X_smote_part = X_sm[mask_for_smote]
                    y_smote_part = y_sm[mask_for_smote]
                    X_not = X_sm[~mask_for_smote]
                    y_not = y_sm[~mask_for_smote]
                    X_res, y_res = sm.fit_resample(X_smote_part, y_smote_part)

                    # Create SMOTE-generated DataFrame with all required columns
                    smote_df = pd.DataFrame(X_res, columns=feature_columns)
                    smote_df['LABEL'] = y_res.values
                    smote_df['SOURCE'] = 'METASTATIC'  # All SMOTE samples are metastatic
                    smote_df['LABEL_NUMERIC'] = pre_data['label_encoder'].transform(y_res.values)

                    # Get the original non-SMOTE data with all columns
                    non_smote_df = train_data_scaled[~mask_for_smote].copy()

                    # Assign unique SAMPLE_IDs to SMOTE-generated samples
                    # Start from a high number to avoid conflicts with existing IDs
                    max_existing_idx = len(train_data_scaled)
                    smote_df['SAMPLE_ID'] = [f"SMOTE_{max_existing_idx + i}" for i in range(len(smote_df))]

                    # Concatenate non-SMOTE and SMOTE data
                    train_data_scaled = pd.concat([non_smote_df, smote_df], ignore_index=True)

                    # Verify all rows have SAMPLE_ID
                    nan_count = train_data_scaled['SAMPLE_ID'].isna().sum()
                    if nan_count > 0:
                        print(f"WARNING: {nan_count} rows still have NaN SAMPLE_ID after SMOTE. Filling them...")
                        nan_mask = train_data_scaled['SAMPLE_ID'].isna()
                        train_data_scaled.loc[nan_mask, 'SAMPLE_ID'] = [
                            f"SMOTE_FILLED_{i}" for i in range(nan_mask.sum())
                        ]

                    print("SMOTE was performed on metastatic classes in the train set.")
                    print(f"  Generated {len(smote_df):,} synthetic samples")
                else:
                    print("  No metastatic classes with >=5 samples found. Skipping SMOTE.")

            val_data_scaled['LABEL_NUMERIC'] = pre_data['label_encoder'].transform(val_data_scaled['LABEL'])
            test_data_scaled['LABEL_NUMERIC'] = pre_data['label_encoder'].transform(test_data_scaled['LABEL'])

            # ========================================================================
            # COMPREHENSIVE DUPLICATE REMOVAL: Remove samples with identical feature values
            # ========================================================================
            print("\n" + "=" * 80)
            print("REMOVING DUPLICATE SAMPLES (identical expression across all 2,547 features)")
            print("=" * 80)

            # Get feature columns for duplicate detection
            feature_cols_for_dedup = [col for col in feature_columns if col in train_data_scaled.columns]

            if feature_cols_for_dedup:
                total_removed = 0
                total_smote_removed = 0
                total_original_removed = 0
                removal_summary = []

                # Track duplicates in each split - use dictionary for efficient updates
                datasets = {
                    'train': train_data_scaled,
                    'val': val_data_scaled,
                    'test': test_data_scaled
                }

                for split_name, data in datasets.items():
                    before_count = len(data)

                    # Find duplicates based on feature columns
                    duplicate_mask = data[feature_cols_for_dedup].duplicated(keep='first')
                    duplicates = data[duplicate_mask]

                    if len(duplicates) > 0:
                        # Categorize where duplicates came from
                        smote_count = 0
                        original_count = 0

                        for idx in duplicates.index:
                            sample_id = data.loc[idx, 'SAMPLE_ID']
                            if pd.notna(sample_id) and str(sample_id).startswith('SMOTE'):
                                smote_count += 1
                            else:
                                original_count += 1

                        # Remove duplicates (keep first occurrence)
                        data_cleaned = data[~duplicate_mask].copy()
                        removed_count = before_count - len(data_cleaned)
                        total_removed += removed_count
                        total_smote_removed += smote_count
                        total_original_removed += original_count

                        # Update the dictionary (which updates the original reference)
                        datasets[split_name] = data_cleaned

                        # Build summary message
                        sources = []
                        if smote_count > 0:
                            sources.append(f"{smote_count} from SMOTE")
                        if original_count > 0:
                            sources.append(f"{original_count} from original data")

                        if len(sources) == 1:
                            source_str = sources[0]
                        elif len(sources) == 2:
                            source_str = f"{sources[0]} and {sources[1]}"
                        else:
                            source_str = "unknown source"

                        removal_summary.append(
                            f"  {split_name.capitalize()}: {removed_count} duplicates ({source_str})")

                        print(f"  {split_name.capitalize()}: Removed {removed_count} duplicate samples ({source_str})")
                    else:
                        print(f"  {split_name.capitalize()}: No duplicates found")

                # Update original variables from dictionary
                train_data_scaled = datasets['train']
                val_data_scaled = datasets['val']
                test_data_scaled = datasets['test']

                print("=" * 80)
                if total_removed > 0:
                    # Build final summary message in requested format
                    if total_smote_removed > 0 and total_original_removed == 0:
                        print(f"TOTAL: {total_removed} duplicate samples removed, all from SMOTE")
                    elif total_smote_removed == 0 and total_original_removed > 0:
                        print(f"TOTAL: {total_removed} duplicate samples removed, all from original data")
                    else:
                        print(
                            f"TOTAL: {total_removed} duplicate samples removed ({total_smote_removed} from SMOTE, {total_original_removed} from original data)")

                    print("\nDetailed breakdown:")
                    for summary in removal_summary:
                        print(summary)
                else:
                    print("No duplicate samples found - all samples have unique expression profiles")
                print("=" * 80 + "\n")
            else:
                print("WARNING: Could not find feature columns for duplicate detection")

            train_data_scaled['SET'] = 'Train'
            val_data_scaled['SET'] = 'Validation'
            test_data_scaled['SET'] = 'Test'

            with open(merged_data_scaled_file, 'wb') as f:
                pickle.dump({
                    'train_data_scaled': train_data_scaled,
                    'val_data_scaled': val_data_scaled,
                    'test_data_scaled': test_data_scaled,
                    'scaler': scaler
                }, f)
            print("Scaled data with SMOTE saved to checkpoint.")

# After all data loading/processing (whether from checkpoint or fresh), ensure we have all required variables
# This section runs regardless of checkpoint status
if 'train_data_scaled' not in locals() or train_data_scaled is None:
    raise ValueError("ERROR: train_data_scaled is not defined. Checkpoint loading or data processing failed.")
if 'val_data_scaled' not in locals() or val_data_scaled is None:
    raise ValueError("ERROR: val_data_scaled is not defined. Checkpoint loading or data processing failed.")
if 'test_data_scaled' not in locals() or test_data_scaled is None:
    raise ValueError("ERROR: test_data_scaled is not defined. Checkpoint loading or data processing failed.")
if 'feature_columns' not in locals() or feature_columns is None:
    raise ValueError("ERROR: feature_columns is not defined. Checkpoint loading or feature selection failed.")
if 'scaler' not in locals() or scaler is None:
    raise ValueError("ERROR: scaler is not defined. Checkpoint loading or scaling failed.")
if 'label_encoder' not in locals() or label_encoder is None:
    raise ValueError("ERROR: label_encoder is not defined. Checkpoint loading or preprocessing failed.")
if 'common_genes' not in locals() or common_genes is None:
    raise ValueError("ERROR: common_genes is not defined. Checkpoint loading or preprocessing failed.")

print("Storing a CSV of how many samples per label in each split for documentation...")
split_label_counts = {}
for nm, dt in zip(['Train', 'Validation', 'Test'],
                  [train_data_scaled, val_data_scaled, test_data_scaled]):
    cts = dt['LABEL'].value_counts().to_dict()
    split_label_counts[nm] = cts

df_split_counts = pd.DataFrame(columns=['Label', 'TrainCount', 'ValCount', 'TestCount'])
all_lbls = (set(split_label_counts['Train'].keys()) |
            set(split_label_counts['Validation'].keys()) |
            set(split_label_counts['Test'].keys()))
for lbl in sorted(list(all_lbls)):
    row_ = {
        'Label': lbl,
        'TrainCount': split_label_counts['Train'].get(lbl, 0),
        'ValCount': split_label_counts['Validation'].get(lbl, 0),
        'TestCount': split_label_counts['Test'].get(lbl, 0)
    }
    # Use pd.concat instead of deprecated append
    df_split_counts = pd.concat([df_split_counts, pd.DataFrame([row_])], ignore_index=True)
df_split_counts.to_csv(os.path.join(evaluation_dir, 'split_label_counts.csv'), index=False)

# ============================================================================
# COMPREHENSIVE UNIT TESTS BEFORE TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE DATA INTEGRITY TESTS BEFORE TRAINING")
print("=" * 80)

test_failed = False
test_errors = []

# Test 1: Verify feature columns exist in scaled data
print("\n[TEST 1] Verifying selected features exist in scaled data...")
missing_features = [f for f in feature_columns if f not in train_data_scaled.columns]
if missing_features:
    test_failed = True
    test_errors.append(
        f"ERROR: {len(missing_features)} selected features missing from scaled data: {missing_features[:5]}...")
    print(f"  ✗ FAILED: {len(missing_features)} features missing")
else:
    print(f"  ✓ PASSED: All {len(feature_columns)} selected features present in scaled data")

# Test 2: Check for data leakage (SAMPLE_ID overlap)
print("\n[TEST 2] Checking for SAMPLE_ID leakage across splits...")
# FIXED: Check for NaN SAMPLE_IDs first - fail if any exist
train_nan_ids = train_data_scaled['SAMPLE_ID'].isna().sum()
val_nan_ids = val_data_scaled['SAMPLE_ID'].isna().sum()
test_nan_ids = test_data_scaled['SAMPLE_ID'].isna().sum()

if train_nan_ids > 0 or val_nan_ids > 0 or test_nan_ids > 0:
    test_failed = True
    test_errors.append(
        f"ERROR: NaN SAMPLE_IDs found - Train: {train_nan_ids}, Val: {val_nan_ids}, Test: {test_nan_ids}")
    print(f"  ✗ FAILED: NaN SAMPLE_IDs detected (Train: {train_nan_ids}, Val: {val_nan_ids}, Test: {test_nan_ids})")
    raise ValueError("CRITICAL: NaN SAMPLE_IDs found. All samples must have valid SAMPLE_IDs.")

train_ids = set(train_data_scaled['SAMPLE_ID'].astype(str))
val_ids = set(val_data_scaled['SAMPLE_ID'].astype(str))
test_ids = set(test_data_scaled['SAMPLE_ID'].astype(str))

train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

if train_val_overlap:
    test_failed = True
    test_errors.append(
        f"ERROR: {len(train_val_overlap)} SAMPLE_IDs overlap between train and val: {list(train_val_overlap)[:3]}...")
    print(f"  ✗ FAILED: {len(train_val_overlap)} SAMPLE_IDs leak between train and val")
else:
    print("  ✓ PASSED: No SAMPLE_ID leakage between train and val")

if train_test_overlap:
    test_failed = True
    test_errors.append(
        f"ERROR: {len(train_test_overlap)} SAMPLE_IDs overlap between train and test: {list(train_test_overlap)[:3]}...")
    print(f"  ✗ FAILED: {len(train_test_overlap)} SAMPLE_IDs leak between train and test")
else:
    print("  ✓ PASSED: No SAMPLE_ID leakage between train and test")

if val_test_overlap:
    test_failed = True
    test_errors.append(
        f"ERROR: {len(val_test_overlap)} SAMPLE_IDs overlap between val and test: {list(val_test_overlap)[:3]}...")
    print(f"  ✗ FAILED: {len(val_test_overlap)} SAMPLE_IDs leak between val and test")
else:
    print("  ✓ PASSED: No SAMPLE_ID leakage between val and test")

# Test 3: Check for DONOR_ID leakage (GTEx samples)
print("\n[TEST 3] Checking for DONOR_ID leakage (GTEx samples only)...")
if 'DONOR_ID' in train_data_scaled.columns:
    # For GTEx samples, DONOR_ID should not be NaN (it's required for splitting)
    gtex_train = train_data_scaled[train_data_scaled['SOURCE'] == 'GTEX']
    gtex_val = val_data_scaled[val_data_scaled['SOURCE'] == 'GTEX']
    gtex_test = test_data_scaled[test_data_scaled['SOURCE'] == 'GTEX']

    if gtex_train['DONOR_ID'].isna().any() or gtex_val['DONOR_ID'].isna().any() or gtex_test['DONOR_ID'].isna().any():
        test_failed = True
        test_errors.append("ERROR: NaN DONOR_IDs found in GTEx samples")
        print("  ✗ FAILED: NaN DONOR_IDs in GTEx samples")

    train_donor_ids = set(gtex_train['DONOR_ID'].astype(str))
    val_donor_ids = set(gtex_val['DONOR_ID'].astype(str))
    test_donor_ids = set(gtex_test['DONOR_ID'].astype(str))

    train_val_donor_overlap = train_donor_ids & val_donor_ids
    train_test_donor_overlap = train_donor_ids & test_donor_ids
    val_test_donor_overlap = val_donor_ids & test_donor_ids

    if train_val_donor_overlap:
        test_failed = True
        test_errors.append(
            f"ERROR: {len(train_val_donor_overlap)} DONOR_IDs overlap between train and val: {list(train_val_donor_overlap)[:3]}...")
        print(f"  ✗ FAILED: {len(train_val_donor_overlap)} DONOR_IDs leak between train and val")
    else:
        print("  ✓ PASSED: No DONOR_ID leakage between train and val")

    if train_test_donor_overlap:
        test_failed = True
        test_errors.append(
            f"ERROR: {len(train_test_donor_overlap)} DONOR_IDs overlap between train and test: {list(train_test_donor_overlap)[:3]}...")
        print(f"  ✗ FAILED: {len(train_test_donor_overlap)} DONOR_IDs leak between train and test")
    else:
        print("  ✓ PASSED: No DONOR_ID leakage between train and test")

    if val_test_donor_overlap:
        test_failed = True
        test_errors.append(
            f"ERROR: {len(val_test_donor_overlap)} DONOR_IDs overlap between val and test: {list(val_test_donor_overlap)[:3]}...")
        print(f"  ✗ FAILED: {len(val_test_donor_overlap)} DONOR_IDs leak between val and test")
    else:
        print("  ✓ PASSED: No DONOR_ID leakage between val and test")
else:
    print("  ✓ PASSED: No DONOR_ID column (not applicable)")

# Test 4: Check for duplicate samples (exact feature matches)
print("\n[TEST 4] Checking for duplicate samples (identical feature values)...")
train_features = train_data_scaled[feature_columns]
val_features = val_data_scaled[feature_columns]
test_features = test_data_scaled[feature_columns]

train_dups = train_features.duplicated().sum()
val_dups = val_features.duplicated().sum()
test_dups = test_features.duplicated().sum()

if train_dups > 0:
    test_failed = True
    test_errors.append(f"ERROR: {train_dups} duplicate samples in training set (identical feature values)")
    print(f"  ✗ FAILED: {train_dups} duplicate samples in train")
else:
    print("  ✓ PASSED: No duplicate samples in train")

if val_dups > 0:
    test_failed = True
    test_errors.append(f"ERROR: {val_dups} duplicate samples in validation set (identical feature values)")
    print(f"  ✗ FAILED: {val_dups} duplicate samples in val")
else:
    print("  ✓ PASSED: No duplicate samples in val")

if test_dups > 0:
    test_failed = True
    test_errors.append(f"ERROR: {test_dups} duplicate samples in test set (identical feature values)")
    print(f"  ✗ FAILED: {test_dups} duplicate samples in test")
else:
    print("  ✓ PASSED: No duplicate samples in test")

# Test 5: Check for NaN/Inf values in feature columns
print("\n[TEST 5] Checking for NaN/Inf values in feature columns...")
train_nan = train_data_scaled[feature_columns].isna().sum().sum()
train_inf = np.isinf(train_data_scaled[feature_columns].values).sum()
val_nan = val_data_scaled[feature_columns].isna().sum().sum()
val_inf = np.isinf(val_data_scaled[feature_columns].values).sum()
test_nan = test_data_scaled[feature_columns].isna().sum().sum()
test_inf = np.isinf(test_data_scaled[feature_columns].values).sum()

if train_nan > 0 or train_inf > 0:
    test_failed = True
    test_errors.append(f"ERROR: Training data has {train_nan} NaN and {train_inf} Inf values in features")
    print(f"  ✗ FAILED: Train has {train_nan} NaN, {train_inf} Inf")
else:
    print("  ✓ PASSED: No NaN/Inf in train features")

if val_nan > 0 or val_inf > 0:
    test_failed = True
    test_errors.append(f"ERROR: Validation data has {val_nan} NaN and {val_inf} Inf values in features")
    print(f"  ✗ FAILED: Val has {val_nan} NaN, {val_inf} Inf")
else:
    print("  ✓ PASSED: No NaN/Inf in val features")

if test_nan > 0 or test_inf > 0:
    test_failed = True
    test_errors.append(f"ERROR: Test data has {test_nan} NaN and {test_inf} Inf values in features")
    print(f"  ✗ FAILED: Test has {test_nan} NaN, {test_inf} Inf")
else:
    print("  ✓ PASSED: No NaN/Inf in test features")

# Test 6: Verify data shapes and sizes
print("\n[TEST 6] Verifying data shapes and sizes...")
train_shape = train_data_scaled[feature_columns].shape
val_shape = val_data_scaled[feature_columns].shape
test_shape = test_data_scaled[feature_columns].shape

if train_shape[1] != len(feature_columns):
    test_failed = True
    test_errors.append(f"ERROR: Train feature count mismatch: {train_shape[1]} != {len(feature_columns)}")
    print(f"  ✗ FAILED: Train feature count mismatch")
else:
    print(f"  ✓ PASSED: Train shape: {train_shape[0]} samples × {train_shape[1]} features")

if val_shape[1] != len(feature_columns):
    test_failed = True
    test_errors.append(f"ERROR: Val feature count mismatch: {val_shape[1]} != {len(feature_columns)}")
    print(f"  ✗ FAILED: Val feature count mismatch")
else:
    print(f"  ✓ PASSED: Val shape: {val_shape[0]} samples × {val_shape[1]} features")

if test_shape[1] != len(feature_columns):
    test_failed = True
    test_errors.append(f"ERROR: Test feature count mismatch: {test_shape[1]} != {len(feature_columns)}")
    print(f"  ✗ FAILED: Test feature count mismatch")
else:
    print(f"  ✓ PASSED: Test shape: {test_shape[0]} samples × {test_shape[1]} features")

# Test 7: Verify required columns exist
print("\n[TEST 7] Verifying required metadata columns exist...")
required_cols = ['LABEL', 'LABEL_NUMERIC', 'SOURCE', 'SAMPLE_ID']
for col in required_cols:
    missing = []
    if col not in train_data_scaled.columns:
        missing.append('train')
    if col not in val_data_scaled.columns:
        missing.append('val')
    if col not in test_data_scaled.columns:
        missing.append('test')

    if missing:
        test_failed = True
        test_errors.append(f"ERROR: Required column '{col}' missing in: {', '.join(missing)}")
        print(f"  ✗ FAILED: Column '{col}' missing in {', '.join(missing)}")
    else:
        print(f"  ✓ PASSED: Column '{col}' present in all splits")

# Test 8: Verify label encoder consistency
print("\n[TEST 8] Verifying label encoder consistency...")
try:
    train_labels_unique = set(train_data_scaled['LABEL'].unique())
    val_labels_unique = set(val_data_scaled['LABEL'].unique())
    test_labels_unique = set(test_data_scaled['LABEL'].unique())
    all_labels = train_labels_unique | val_labels_unique | test_labels_unique

    # Check if label encoder can handle all labels
    encoded = label_encoder.transform(list(all_labels))
    decoded = label_encoder.inverse_transform(encoded)

    if set(decoded) == all_labels:
        print(f"  ✓ PASSED: Label encoder handles all {len(all_labels)} unique labels")
    else:
        test_failed = True
        test_errors.append("ERROR: Label encoder cannot handle all labels")
        print("  ✗ FAILED: Label encoder inconsistency")
except Exception as e:
    test_failed = True
    test_errors.append(f"ERROR: Label encoder test failed: {e}")
    print(f"  ✗ FAILED: Label encoder error: {e}")

# Test 9: Verify feature selection was applied (scaled data should only have selected features used)
print("\n[TEST 9] Verifying feature selection was applied correctly...")
# Check that we're using only the selected features for training
if len(feature_columns) <= len(common_genes):
    print(f"  ✓ PASSED: Feature selection applied ({len(feature_columns)}/{len(common_genes)} genes selected)")
else:
    test_failed = True
    test_errors.append(f"ERROR: Feature selection issue: {len(feature_columns)} > {len(common_genes)}")
    print(f"  ✗ FAILED: Feature count mismatch")

# Test 10: Verify data is scaled (mean ~0, std ~1 for train set)
print("\n[TEST 10] Verifying data scaling (StandardScaler applied)...")
train_feat_mean = train_data_scaled[feature_columns].mean().abs().mean()
train_feat_std = train_data_scaled[feature_columns].std().mean()

# StandardScaler should give mean ~0 and std ~1
# After SMOTE, mean may be slightly non-zero (synthetic samples), so we allow more tolerance
# Mean tolerance: < 0.1 (allows for SMOTE effects)
# Std tolerance: 0.9-1.1 (allows for slight variations)
if train_feat_mean < 0.1 and 0.9 <= train_feat_std <= 1.1:
    print(f"  ✓ PASSED: Data properly scaled (mean≈{train_feat_mean:.3f}, std≈{train_feat_std:.3f})")
else:
    test_failed = True
    test_errors.append(f"ERROR: Data may not be properly scaled (mean={train_feat_mean:.3f}, std={train_feat_std:.3f})")
    print(f"  ✗ FAILED: Scaling issue (mean={train_feat_mean:.3f}, std={train_feat_std:.3f})")

# Final summary
print("\n" + "=" * 80)
if test_failed:
    print("✗✗✗ DATA INTEGRITY TESTS FAILED ✗✗✗")
    print("\nErrors found:")
    for error in test_errors:
        print(f"  - {error}")
    print("\n" + "=" * 80)
    raise ValueError("DATA INTEGRITY TESTS FAILED. Please fix the issues above before training models.")
else:
    print("✓✓✓ ALL DATA INTEGRITY TESTS PASSED ✓✓✓")
    print("Data is ready for training.")
    print("=" * 80 + "\n")

print("Defining and training models...")
fcols = feature_columns

print("Attempting to create subdirectories for SNN and CAE evaluations...")
snn_eval_dir = os.path.join(evaluation_dir, "SNN_Evaluation")
cae_eval_dir = os.path.join(evaluation_dir, "CAE_Evaluation")
os.makedirs(snn_eval_dir, exist_ok=True)
os.makedirs(cae_eval_dir, exist_ok=True)


# ------------------ SNN & CAE Training and Evaluation (Modified) ------------------

def evaluate_snn(model, generator, output_dir):
    """
    Evaluate Siamese neural network on test set with pair classification metrics.

    Computes binary classification performance metrics for the Siamese network
    on a held-out test set. Generates visualizations including confusion matrix
    and ROC curve to assess model performance on pair similarity prediction.

    Args:
        model (tf.keras.Model): Trained Siamese network model
        generator (PairGenerator): Data generator providing test pairs and labels
        output_dir (str): Directory to save evaluation plots

    Returns:
        None: Saves evaluation plots to output_dir

    Notes:
        - Predictions are binarized at threshold 0.5
        - Generates confusion matrix and ROC curve side-by-side
        - Saves figure as 'snn_confusion_and_roc.png' in output_dir
        - Uses binary classification metrics (0=dissimilar, 1=similar pairs)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import numpy as np
    import seaborn as sns
    import os

    # Collect predictions
    all_labels = []
    all_preds = []
    for idx in range(len(generator)):
        (X1_batch, X2_batch), y_batch = generator[idx]
        preds_batch = model.predict([X1_batch, X2_batch], verbose=0).ravel()
        all_labels.extend(y_batch)
        all_preds.extend(preds_batch)

    all_labels = np.array(all_labels, dtype=int)
    all_preds = np.array(all_preds)

    # Binarize predictions at threshold=0.5
    threshold = 0.5
    bin_preds = (all_preds >= threshold).astype(int)

    # Confusion Matrix (binary)
    cm = confusion_matrix(all_labels, bin_preds, labels=[0, 1])

    # ROC curve, AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc_val = auc(fpr, tpr)

    # 1x2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Confusion Matrix on ax[0]
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Pred:0', 'Pred:1'], yticklabels=['True:0', 'True:1'],
                ax=ax[0])
    ax[0].set_title("SNN Confusion Matrix (Test)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # (B) ROC on ax[1]
    ax[1].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc_val:.3f}")
    ax[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax[1].set_title("SNN ROC (Test)")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'snn_confusion_and_roc.png'), dpi=300)
    plt.close(fig)


# ------------------ SNN Training (40 epochs total) ------------------
# Check if SNN model exists - if so, load it; otherwise train new model
if not os.path.exists(snn_model_file):
    print("Training SNN with overrepresented metastatic pairs...")
    with timer("SNN Training (40 epochs)"):
        # 1) Create training/validation pairs (pairwise, 0 or 1 label)
        train_pairs_snn, train_labels_snn = create_pairs_by_logic(
            train_data_scaled,
            max_pairs=max_pairs,
            overrepresent_metastatic=True
            # You can add any 'stratify' logic if needed in create_pairs_by_logic
        )
    val_pairs_snn, val_labels_snn = create_pairs_by_logic(
        val_data_scaled,
        max_pairs=max_pairs,
        overrepresent_metastatic=False
    )

    # 2) Generators
    train_gen_snn = PairGenerator(
        train_pairs_snn, train_labels_snn,
        train_data_scaled, fcols,
        batch_size=64, augment=True
    )
    val_gen_snn = PairGenerator(
        val_pairs_snn, val_labels_snn,
        val_data_scaled, fcols,
        batch_size=64
    )

    # 3) Build SNN model
    snn_base = create_modified_siamese_network(len(fcols))
    inpA = layers.Input(shape=(len(fcols),))
    inpB = layers.Input(shape=(len(fcols),))
    outA = snn_base(inpA)
    outB = snn_base(inpB)
    dist_layer = layers.Lambda(lambda xx: K.abs(xx[0] - xx[1]))([outA, outB])
    out = layers.Dense(1, activation='sigmoid')(dist_layer)
    snn_model = models.Model([inpA, inpB], out)
    snn_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    # 4) Train in four phases => total 40 epochs
    init_epochs = 10
    uncert_epochs = 10
    overconf_epochs = 10
    final_epochs = 10

    # --- Phase A: Initial (epochs=10) ---
    hist_snn = snn_model.fit(
        train_gen_snn,
        validation_data=val_gen_snn,
        epochs=init_epochs,
        verbose=1
    )

    # --- Phase B: Uncertainty HEM (epochs=10) ---
    for _ in range(uncert_epochs):
        hidx = get_hard_examples_snn(
            snn_model, train_gen_snn,
            top_k_percent=0.15,
            min_num_samples=64 * 50,
            mode='uncertainty'
        )
        hgen_snn = PairGenerator(
            train_pairs_snn, train_labels_snn,
            train_data_scaled, fcols,
            batch_size=64, augment=True,
            indices=hidx
        )
        one_ep_hist = snn_model.fit(
            hgen_snn,
            validation_data=val_gen_snn,
            epochs=1,
            verbose=1
        )
        for k, v in one_ep_hist.history.items():
            hist_snn.history[k].extend(v)

    # --- Phase C: Overconfidence HEM (epochs=10) ---
    for _ in range(overconf_epochs):
        hidx = get_hard_examples_snn(
            snn_model, train_gen_snn,
            top_k_percent=0.15,
            min_num_samples=64 * 50,
            mode='overconfident'
        )
        hgen_snn = PairGenerator(
            train_pairs_snn, train_labels_snn,
            train_data_scaled, fcols,
            batch_size=64, augment=True,
            indices=hidx
        )
        one_ep_hist = snn_model.fit(
            hgen_snn,
            validation_data=val_gen_snn,
            epochs=1,
            verbose=1
        )
        for k, v in one_ep_hist.history.items():
            hist_snn.history[k].extend(v)

    # --- Final Phase: last 10 epochs ---
    print("Final 10 epochs of convergence for SNN...")
    final_hist = snn_model.fit(
        train_gen_snn,
        validation_data=val_gen_snn,
        epochs=final_epochs,
        verbose=1
    )
    for k, v in final_hist.history.items():
        hist_snn.history[k].extend(v)

    # Save model
    snn_model.save(snn_model_file)

    # Plot with 3 dotted lines => red, yellow, black
    plot_train_val_loss_two_blocks(
        history=hist_snn,
        uncertainty_start=init_epochs,
        overconfidence_start=init_epochs + uncert_epochs,
        end_hem=init_epochs + uncert_epochs + overconf_epochs,
        plot_title="SNN Train vs Val Loss",
        plot_filename=os.path.join(snn_eval_dir, "snn_loss.png")
    )

else:
    print("SNN model file exists. Loading model...")
    try:
        snn_model = tf.keras.models.load_model(snn_model_file)
        print("  SNN model loaded successfully")

        # EXTRACT BASE NETWORK - CRITICAL FIX
        # SNN architecture: [Input, Input] → [Base, Base] → Lambda → Dense
        # Base network is typically a shared weight layer (Model or Sequential)
        snn_base_network = None
        for layer in snn_model.layers:
            if 'model' in layer.name.lower() and hasattr(layer, 'layers'):
                snn_base_network = layer
                print(f"  Extracted SNN base network: {layer.name}")
                break

        if snn_base_network is None:
            # Fallback: try to extract from layer index or rebuild
            if len(snn_model.layers) >= 3:
                # Try layers[2] as fallback (common SNN structure)
                potential_base = snn_model.layers[2]
                if hasattr(potential_base, 'layers') or hasattr(potential_base, 'get_weights'):
                    snn_base_network = potential_base
                    print(f"  Extracted SNN base network from layer index 2: {potential_base.name}")
                else:
                    # Last resort: rebuild base network from model structure
                    print("  WARNING: Could not auto-extract base network. Rebuilding...")
                    snn_base_network = create_modified_siamese_network(len(fcols))
                    # Try to copy weights if possible
                    try:
                        if len(snn_model.layers) >= 3 and hasattr(snn_model.layers[2], 'layers'):
                            source_layers = snn_model.layers[2].layers
                            target_layers = snn_base_network.layers
                            for i, (src, tgt) in enumerate(zip(source_layers, target_layers)):
                                if hasattr(src, 'get_weights') and hasattr(tgt, 'set_weights'):
                                    tgt.set_weights(src.get_weights())
                    except Exception as e:
                        print(f"  WARNING: Could not copy weights: {e}")
            else:
                raise ValueError("ERROR: SNN model structure unexpected. Cannot extract base network.")
    except Exception as e:
        print(f"ERROR: Failed to load SNN model: {e}")
        print("Deleting corrupted model file and retraining...")
        if os.path.exists(snn_model_file):
            os.remove(snn_model_file)
        # Don't continue with None - force retrain
        print("CRITICAL: SNN model loading failed. Retraining from scratch...")
        raise RuntimeError(f"SNN model loading failed: {e}. Model file deleted. Please retrain.")


# ------------------ CAE (Dual-Loss: Contrastive + Reconstruction) ------------------

class PairGeneratorCAE(Sequence):
    """
    Specialized generator for CAE with dual outputs:
      - Output #1: distz (contrastive)
      - Output #2 and #3: reconstructions of X1, X2

    So, the model expects y = [ contrastive_label, X1, X2 ] to match:
      outputs = [ distz, decA, decB ].
    """

    def __init__(self,
                 pairs,
                 labels,
                 data,
                 feature_columns,
                 batch_size=64,
                 augment=False,
                 indices=None):
        if indices is not None:
            self.pairs = pairs[indices]
            self.labels = labels[indices]
        else:
            self.pairs = pairs
            self.labels = labels

        self.data = data.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.indices = np.arange(len(self.pairs))
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_pairs = self.pairs[batch_indices]
        batch_labels = self.labels[batch_indices]

        X1_batch = self.data.iloc[batch_pairs[:, 0]][self.feature_columns].values.astype(np.float32)
        X2_batch = self.data.iloc[batch_pairs[:, 1]][self.feature_columns].values.astype(np.float32)

        if self.augment:
            X1_batch += np.random.normal(0, 0.01, X1_batch.shape)
            X2_batch += np.random.normal(0, 0.01, X2_batch.shape)

        y_contrast = batch_labels.reshape(-1, 1).astype('float32')

        return ([X1_batch, X2_batch],
                [y_contrast, X1_batch, X2_batch])


def create_dual_loss_cae(input_dim):
    """
    Builds a CAE model with 3 outputs:
      - distz: contrastive distance
      - decA: reconstruction of input A
      - decB: reconstruction of input B
    Losses: [contrastive, mse, mse]
    """
    initializer = tf.keras.initializers.HeNormal(seed=SEED)

    # Encoder
    inp_enc = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(inp_enc)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    encoded = layers.Dense(64, activation='tanh', kernel_initializer=initializer,
                           kernel_regularizer=regularizers.l2(1e-4))(x)
    encoded = layers.Lambda(lambda xx: K.l2_normalize(xx, axis=1))(encoded)
    encoder = models.Model(inp_enc, encoded, name='encoder')

    # Decoder
    inp_dec = layers.Input(shape=(64,))
    xd = layers.Dense(128, activation='relu', kernel_initializer=initializer,
                      kernel_regularizer=regularizers.l2(1e-4))(inp_dec)
    xd = layers.BatchNormalization()(xd)
    xd = layers.Dropout(0.5)(xd)
    xd = layers.Dense(256, activation='relu', kernel_initializer=initializer,
                      kernel_regularizer=regularizers.l2(1e-4))(xd)
    xd = layers.BatchNormalization()(xd)
    dec_out = layers.Dense(input_dim, activation='linear',
                           kernel_initializer=initializer)(xd)
    decoder = models.Model(inp_dec, dec_out, name='decoder')

    # Contrastive distance function
    def eucl_dist(xx):
        x1, x2 = xx
        sq = K.sum(K.square(x1 - x2), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sq, K.epsilon()))

    def contrastive_loss_fn(y_true, dist):
        margin = 1.0
        y_true = K.cast(y_true, dist.dtype)
        sq = K.square(dist)
        margin_sq = K.square(K.maximum(margin - dist, 0))
        return K.mean(y_true * sq + (1. - y_true) * margin_sq)

    # Combined CAE model
    inpA = layers.Input(shape=(input_dim,), name='cae_inputA')
    inpB = layers.Input(shape=(input_dim,), name='cae_inputB')
    encA = encoder(inpA)
    encB = encoder(inpB)
    distz = layers.Lambda(eucl_dist)([encA, encB])

    decA = decoder(encA)
    decB = decoder(encB)

    # 3 outputs: [distz, decA, decB]
    cae_model = models.Model([inpA, inpB], [distz, decA, decB], name='cae_dual')

    # Compile with multi-loss
    cae_model.compile(
        loss=[contrastive_loss_fn, 'mse', 'mse'],
        loss_weights=[1.0, 1.0, 1.0],  # can tune weighting if desired
        optimizer=optimizers.Adam(learning_rate=1e-4)
    )

    return cae_model, encoder, decoder


# ------------------ CAE Training (60 epochs total) ------------------
# Check if CAE models exist - if so, load them; otherwise train new models
if not os.path.exists(cae_encoder_file) or not os.path.exists(cae_decoder_file):
    print("Training CAE (Dual-Loss) with overrepresented metastatic pairs...")
    with timer("CAE Training"):
        # 1) Create train/val pairs
        train_pairs_cae, train_labels_cae = create_pairs_by_logic(
            train_data_scaled,
            max_pairs=max_pairs,
            overrepresent_metastatic=True
        )
    val_pairs_cae, val_labels_cae = create_pairs_by_logic(
        val_data_scaled,
        max_pairs=max_pairs,
        overrepresent_metastatic=False
    )

    # 2) Generators for dual-loss CAE
    train_gen_cae = PairGeneratorCAE(
        train_pairs_cae, train_labels_cae,
        train_data_scaled, fcols,
        batch_size=64, augment=True
    )
    val_gen_cae = PairGeneratorCAE(
        val_pairs_cae, val_labels_cae,
        val_data_scaled, fcols,
        batch_size=64
    )

    # 3) Build the dual-loss CAE
    cae_model, cae_encoder, cae_decoder = create_dual_loss_cae(len(fcols))

    # 4) Four-phase training => total 40 epochs
    init_epochs_cae = 10
    uncert_epochs_cae = 10
    overconf_epochs_cae = 10
    final_epochs_cae = 30

    # --- Phase A ---
    hist_cae = cae_model.fit(
        train_gen_cae,
        validation_data=val_gen_cae,
        epochs=init_epochs_cae,
        verbose=1
    )

    # --- Phase B: Uncertainty HEM ---
    for _ in range(uncert_epochs_cae):
        hidx = get_hard_examples_cae(
            cae_model, train_gen_cae,
            top_k_percent=0.15,
            min_num_samples=64 * 50,
            margin=1.0,
            mode='uncertainty'
        )
        hgen_cae = PairGeneratorCAE(
            train_pairs_cae, train_labels_cae,
            train_data_scaled, fcols,
            batch_size=64, augment=True,
            indices=hidx
        )
        one_ep_hist = cae_model.fit(
            hgen_cae,
            validation_data=val_gen_cae,
            epochs=1,
            verbose=1
        )
        for k, v in one_ep_hist.history.items():
            hist_cae.history[k].extend(v)

    # --- Phase C: Overconfident HEM ---
    for _ in range(overconf_epochs_cae):
        hidx = get_hard_examples_cae(
            cae_model, train_gen_cae,
            top_k_percent=0.15,
            min_num_samples=64 * 50,
            margin=1.0,
            mode='overconfident'
        )
        hgen_cae = PairGeneratorCAE(
            train_pairs_cae, train_labels_cae,
            train_data_scaled, fcols,
            batch_size=64, augment=True,
            indices=hidx
        )
        one_ep_hist = cae_model.fit(
            hgen_cae,
            validation_data=val_gen_cae,
            epochs=1,
            verbose=1
        )
        for k, v in one_ep_hist.history.items():
            hist_cae.history[k].extend(v)

    # --- Phase D: final epochs ---
    print("Final 10 epochs of CAE training...")
    final_hist_cae = cae_model.fit(
        train_gen_cae,
        validation_data=val_gen_cae,
        epochs=final_epochs_cae,
        verbose=1
    )
    for k, v in final_hist_cae.history.items():
        hist_cae.history[k].extend(v)

    # Save
    cae_model.save(os.path.join(new_output_dir, "cae_autoencoder_dual_loss.h5"))
    cae_encoder.save(cae_encoder_file)
    cae_decoder.save(cae_decoder_file)

    # Plot training curve
    plot_train_val_loss_two_blocks(
        history=hist_cae,
        uncertainty_start=init_epochs_cae,
        overconfidence_start=init_epochs_cae + uncert_epochs_cae,
        end_hem=init_epochs_cae + uncert_epochs_cae + overconf_epochs_cae,
        plot_title="CAE (Dual-Loss) Train vs Val Loss",
        plot_filename=os.path.join(cae_eval_dir, "cae_loss.png")
    )

else:
    print("CAE encoder/decoder files exist. Loading models...")
    try:
        cae_encoder = tf.keras.models.load_model(cae_encoder_file)
        cae_decoder = tf.keras.models.load_model(cae_decoder_file)
        print("  CAE models loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load CAE models: {e}")
        print("Deleting corrupted model files and retraining...")
        if os.path.exists(cae_encoder_file):
            os.remove(cae_encoder_file)
        if os.path.exists(cae_decoder_file):
            os.remove(cae_decoder_file)
        # Don't continue with None - force retrain
        print("CRITICAL: CAE model loading failed. Retraining from scratch...")
        raise RuntimeError(f"CAE model loading failed: {e}. Model files deleted. Please retrain.")

# -------------- Evaluate SNN & CAE on Test --------------
print("Evaluating SNN on held-out test set (pairs) for direct classification...")

test_pairs_snn, test_labels_snn = create_pairs_by_logic(
    test_data_scaled,
    max_pairs=50000,
    overrepresent_metastatic=False
)
test_gen_snn = PairGenerator(
    test_pairs_snn,
    test_labels_snn,
    test_data_scaled,
    feature_columns=fcols,
    batch_size=64
)

eval_snn_dir = os.path.join(snn_eval_dir, "SNN_Test")
os.makedirs(eval_snn_dir, exist_ok=True)
evaluate_snn(snn_model, test_gen_snn, eval_snn_dir)

print("Evaluating CAE by reconstructing top-2 best reconstructions from the held-out test set...")
eval_cae_dir = os.path.join(cae_eval_dir, "CAE_Test")
os.makedirs(eval_cae_dir, exist_ok=True)
evaluate_cae_autoencoder(
    cae_encoder,
    cae_decoder,
    test_data_scaled,
    fcols,
    eval_cae_dir
)

# ------------------ End of Modified SNN & CAE Section ------------------

# -------------- Generate embeddings & meta-learning --------------
print("Generating embeddings for meta-learning...")

###############################################################################
# 1) Prepare data for embeddings and meta-learning
###############################################################################
# CRITICAL FIX: Use ALL training labels (no filtering based on test set)
# This prevents data leakage - the test set structure should not influence training
print("\n" + "=" * 80)
print("GENERATING EMBEDDINGS FOR BASE CLASSIFIERS AND META-LEARNER")
print("=" * 80)
print("Using ALL training labels (no filtering based on test set)")

# Safety check: Ensure models are loaded
if 'snn_base_network' not in locals() or snn_base_network is None:
    if 'snn_model' in locals() and snn_model is not None:
        # Try to extract base network using the same logic as loading
        snn_base_network = None
        for layer in snn_model.layers:
            if 'model' in layer.name.lower() and hasattr(layer, 'layers'):
                snn_base_network = layer
                print(f"  Extracted SNN base network: {layer.name}")
                break
        if snn_base_network is None and len(snn_model.layers) >= 3:
            potential_base = snn_model.layers[2]
            if hasattr(potential_base, 'layers') or hasattr(potential_base, 'get_weights'):
                snn_base_network = potential_base
                print(f"  Extracted SNN base network from layer index 2: {potential_base.name}")
        if snn_base_network is None:
            raise ValueError("ERROR: SNN base network could not be extracted. Cannot generate embeddings.")
    else:
        raise ValueError("ERROR: SNN model not loaded. Cannot generate embeddings.")
if 'cae_encoder' not in locals() or cae_encoder is None:
    raise ValueError("ERROR: CAE encoder not loaded. Cannot generate embeddings.")

# Safety check: Ensure fcols is defined
if 'fcols' not in locals() or fcols is None:
    if 'feature_columns' in locals():
        fcols = feature_columns
    else:
        raise ValueError("ERROR: feature_columns not defined. Cannot generate embeddings.")


print("\n" + "=" * 80)
print("GENERATING EMBEDDINGS FOR BASE CLASSIFIERS AND META-LEARNER")
print("=" * 80)
print("CRITICAL: Using ALL training labels (not filtering based on test set)")
print("          This prevents data leakage and ensures fair evaluation")
print("=" * 80)

train_data_scaled_for_emb = train_data_scaled.copy()
val_data_scaled_for_emb = val_data_scaled.copy()
test_data_scaled_for_emb = test_data_scaled.copy()

# Generate embeddings for ALL data
print("Generating SNN embeddings...")
snn_emb_train = snn_base_network.predict(train_data_scaled_for_emb[fcols].values, verbose=0)
snn_emb_val = snn_base_network.predict(val_data_scaled_for_emb[fcols].values, verbose=0)
snn_emb_test = snn_base_network.predict(test_data_scaled_for_emb[fcols].values, verbose=0)

print("Generating CAE embeddings...")
cae_emb_train = cae_encoder.predict(train_data_scaled_for_emb[fcols].values, verbose=0)
cae_emb_val = cae_encoder.predict(val_data_scaled_for_emb[fcols].values, verbose=0)
cae_emb_test = cae_encoder.predict(test_data_scaled_for_emb[fcols].values, verbose=0)

# Combined embeddings => shape (None, 128) if each is 64D
combined_train = np.concatenate([snn_emb_train, cae_emb_train], axis=1)
combined_val = np.concatenate([snn_emb_val, cae_emb_val], axis=1)
combined_test = np.concatenate([snn_emb_test, cae_emb_test], axis=1)

print(f"  Train embeddings: {combined_train.shape}")
print(f"  Val embeddings: {combined_val.shape}")
print(f"  Test embeddings: {combined_test.shape}")
print("=" * 80 + "\n")

###############################################################################
# 3) NESTED SPLIT: Re-split train+val (80%) into 80-20 for base/meta training
###############################################################################
print("\n" + "=" * 80)
print("NESTED SPLIT FOR META-LEARNING")
print("=" * 80)
print("Original splits: Train+Val (80%) → Test (20%)")
print("Re-splitting Train+Val (80%) into: Base/Meta Train (80%) → Meta Val (20%)")
print("Final evaluation on original Test (20%)")
print("=" * 80)


combined_train_val = np.concatenate([combined_train, combined_val], axis=0)
train_val_labels = np.concatenate([
    train_data_scaled_for_emb['LABEL'].values,
    val_data_scaled_for_emb['LABEL'].values
], axis=0)
train_val_data = pd.concat([
    train_data_scaled_for_emb.reset_index(drop=True),
    val_data_scaled_for_emb.reset_index(drop=True)
], ignore_index=True)

meta_label_encoder = LabelEncoder()
train_val_labels_unique = np.unique(train_val_labels)
meta_label_encoder.fit(train_val_labels_unique)
met_class_labels = meta_label_encoder.classes_

print(f"Base classifiers/Meta-learner will predict {len(met_class_labels)} labels (from training data)")
print(f"  Training labels: {sorted(met_class_labels)[:5]}... (showing first 5 of {len(met_class_labels)})")

test_labels_set = set(test_data_scaled_for_emb['LABEL'].unique())
train_labels_set = set(train_val_labels_unique)
labels_only_in_test = test_labels_set - train_labels_set
labels_only_in_train = train_labels_set - test_labels_set

if len(labels_only_in_test) > 0:
    print(f"\n  WARNING: {len(labels_only_in_test)} labels appear in test but NOT in training:")
    print(f"        {sorted(list(labels_only_in_test))[:3]}...")
    print(f"        These will be handled as 'unknown' during evaluation")
if len(labels_only_in_train) > 0:
    print(f"\n  NOTE: {len(labels_only_in_train)} labels appear in training but NOT in test:")
    print(f"        {sorted(list(labels_only_in_train))[:3]}...")
    print(f"        Model will learn these labels but won't be evaluated on them")

# Re-split train+val into 80-20 (stratified by label)
if len(combined_train_val) == 0:
    raise ValueError("ERROR: No data available for nested split! Combined train+val is empty.")
if len(train_val_labels) == 0:
    raise ValueError("ERROR: No labels available for nested split!")

y_train_val_encoded = meta_label_encoder.transform(train_val_labels)

unique_labels_in_split = np.unique(y_train_val_encoded)
min_samples_per_class = min([np.sum(y_train_val_encoded == lbl) for lbl in unique_labels_in_split])
if min_samples_per_class < 2:
    print(f"WARNING: Some classes have <2 samples. Stratification may fail. Using shuffle=True instead.")
    stratify_param = None
else:
    stratify_param = y_train_val_encoded

combined_base_train, combined_meta_val, y_base_train, y_meta_val, train_val_base, train_val_meta = train_test_split(
    combined_train_val,
    y_train_val_encoded,
    train_val_data,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_param
)

if len(combined_base_train) == 0:
    raise ValueError("ERROR: Base training set is empty after nested split!")
if len(combined_meta_val) == 0:
    raise ValueError("ERROR: Meta validation set is empty after nested split!")
if len(combined_test) == 0:
    raise ValueError("ERROR: Test set is empty!")


train_pct = (len(combined_base_train) / len(combined_train_val) * 100) if len(combined_train_val) > 0 else 0
val_pct = (len(combined_meta_val) / len(combined_train_val) * 100) if len(combined_train_val) > 0 else 0

print(f"Base/Meta Train: {len(combined_base_train):,} samples ({train_pct:.1f}%)")
print(f"Meta Validation: {len(combined_meta_val):,} samples ({val_pct:.1f}%)")
print(f"Final Test (held out): {len(combined_test):,} samples")
print("=" * 80 + "\n")

###############################################################################
# 4) Train & Store Base Classifiers on the 80% split (base_train)
#    so each classifier sees 128-D input => outputs (#labels) prob.
#    GridSearchCV tuning (4 hyperparameters per classifier, 5-fold CV)
###############################################################################
base_classifiers_file = os.path.join(evaluation_dir, "trained_base_classifiers.pkl")
base_learners_csv_file = os.path.join(evaluation_dir, "base_learners_matrices.csv")

# Check if base classifiers already exist
if os.path.exists(base_classifiers_file):
    print("Base classifiers checkpoint found. Loading...")
    with open(base_classifiers_file, 'rb') as f:
        met_classifiers = pickle.load(f)
    print("Base classifiers loaded from checkpoint.")

    # Try to load CSV if it exists
    if os.path.exists(base_learners_csv_file):
        print(f"Base learners tuning results CSV found: {base_learners_csv_file}")
    else:
        print("WARNING: Base classifiers loaded but tuning CSV not found.")
else:
    print("Training base classifiers on 80% of train+val data...")
    print("Performing GridSearchCV tuning (4 hyperparameters per classifier, 5-fold CV)...")
    with timer("Base Classifier Training & Tuning"):
        if len(combined_base_train) == 0:
            raise ValueError("ERROR: Base training set is empty! Cannot train classifiers.")
        if len(y_base_train) == 0:
            raise ValueError("ERROR: Base training labels are empty!")
        if len(np.unique(y_base_train)) < 2:
            raise ValueError(
                f"ERROR: Need at least 2 classes for classification. Found {len(np.unique(y_base_train))} class(es).")

    # Define parameter grids for each classifier (4 hyperparameters each)
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],  
            'solver': ['saga'],  
            'max_iter': [500, 1000]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'p': [1, 2]  
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3] 
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
    }

    # Base classifier definitions (will be replaced by tuned versions)
    base_classifier_definitions = {
        'LogisticRegression': LogisticRegression(random_state=SEED),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=SEED),
        'SVM': SVC(probability=True, random_state=SEED),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                 verbosity=0, objective='multi:softprob', random_state=SEED)
    }

    # Perform GridSearchCV for each classifier
    met_classifiers = {}
    cv_folds = 5  # Increased from 3 to 5 for better reliability

    # Store all tuning results for CSV
    tuning_results = []

    print("\n" + "=" * 80)
    print("GRIDSEARCHCV TUNING FOR BASE CLASSIFIERS")
    print("=" * 80)

    for c_name, base_clf in base_classifier_definitions.items():
        print(f"\nTuning {c_name}...")

        # Special handling for XGBoost
        if c_name == 'XGBoost':
            base_clf.set_params(num_class=len(met_class_labels))

        # Get parameter grid
        param_grid = param_grids[c_name]

        # Perform grid search with 5-fold CV
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,  # Use all available cores
            verbose=1
        )

        try:
            grid_search.fit(combined_base_train, y_base_train)
            met_classifiers[c_name] = grid_search.best_estimator_

            # Store results
            for params, mean_score, std_score in zip(
                    grid_search.cv_results_['params'],
                    grid_search.cv_results_['mean_test_score'],
                    grid_search.cv_results_['std_test_score']
            ):
                result_row = {
                    'Classifier': c_name,
                    'C': params.get('C', np.nan),
                    'penalty': params.get('penalty', np.nan),
                    'solver': params.get('solver', np.nan),
                    'max_iter': params.get('max_iter', np.nan),
                    'n_neighbors': params.get('n_neighbors', np.nan),
                    'weights': params.get('weights', np.nan),
                    'metric': params.get('metric', np.nan),
                    'p': params.get('p', np.nan),
                    'n_estimators': params.get('n_estimators', np.nan),
                    'max_depth': params.get('max_depth', np.nan),
                    'min_samples_split': params.get('min_samples_split', np.nan),
                    'min_samples_leaf': params.get('min_samples_leaf', np.nan),
                    'kernel': params.get('kernel', np.nan),
                    'gamma': params.get('gamma', np.nan),
                    'degree': params.get('degree', np.nan),
                    'learning_rate': params.get('learning_rate', np.nan),
                    'subsample': params.get('subsample', np.nan),
                    'Mean_CV_Score': mean_score,
                    'Std_CV_Score': std_score,
                    'Is_Best': (params == grid_search.best_params_)
                }
                tuning_results.append(result_row)

            print(f"  Best parameters: {grid_search.best_params_}")
            print(
                f"  Best CV score: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f})")
        except Exception as e:
            print(f"  WARNING: GridSearchCV failed for {c_name}: {str(e)}")
            print(f"  Falling back to default parameters...")
            # Fallback to default classifier
            if c_name == 'XGBoost':
                base_clf.set_params(num_class=len(met_class_labels))
            met_classifiers[c_name] = base_clf
            met_classifiers[c_name].fit(combined_base_train, y_base_train)

            # Store default result
            result_row = {
                'Classifier': c_name,
                'Mean_CV_Score': np.nan,
                'Std_CV_Score': np.nan,
                'Is_Best': True,
                'Note': 'Default parameters (GridSearchCV failed)'
            }
            tuning_results.append(result_row)

    print("\n" + "=" * 80)
    print("GridSearchCV tuning complete for all base classifiers")
    print("=" * 80 + "\n")

    # Save base classifiers
    with open(base_classifiers_file, "wb") as f:
        pickle.dump(met_classifiers, f)
    print("Base classifiers trained & stored (on 80% of train+val data).")

# Save tuning results to CSV
if tuning_results:
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(base_learners_csv_file, index=False)
    print(f"Base learners tuning results saved to: {base_learners_csv_file}")

###############################################################################
# 5) Build meta-features from base classifiers
#    - Meta-features for meta-learner training: CV-based from base_train (80%)
#    - Meta-features for meta-learner validation: direct from meta_val (20% of train+val)
#    - Meta-features for final evaluation: direct from test (20% held out)
###############################################################################
if not met_classifiers or len(met_classifiers) == 0:
    raise ValueError("ERROR: Base classifiers not trained! Cannot generate meta-features.")

print("\n" + "=" * 80)
print("GENERATING META-FEATURES")
print("=" * 80)
print("Using cross-validation for base_train to avoid data leakage...")

# Use CV for base_train to avoid leakage 
meta_feats_base_train = get_meta_features_cv(
    met_classifiers, combined_base_train, y_base_train, cv_folds=5, random_state=SEED
)

# Direct prediction for meta_val and test (these are truly held-out)
meta_feats_meta_val = get_meta_features(met_classifiers, combined_meta_val)
meta_feats_test = get_meta_features(met_classifiers, combined_test)

print(f"  Base train meta-features (CV): {meta_feats_base_train.shape}")
print(f"  Meta val meta-features: {meta_feats_meta_val.shape}")
print(f"  Test meta-features: {meta_feats_test.shape}")
print("=" * 80 + "\n")

if meta_feats_base_train.shape[0] != len(combined_base_train):
    raise ValueError(
        f"ERROR: Meta-features shape mismatch. Expected {len(combined_base_train)} samples, got {meta_feats_base_train.shape[0]}")
if meta_feats_meta_val.shape[0] != len(combined_meta_val):
    raise ValueError(
        f"ERROR: Meta-features shape mismatch. Expected {len(combined_meta_val)} samples, got {meta_feats_meta_val.shape[0]}")
if meta_feats_test.shape[0] != len(combined_test):
    raise ValueError(
        f"ERROR: Meta-features shape mismatch. Expected {len(combined_test)} samples, got {meta_feats_test.shape[0]}")

test_labels_raw = test_data_scaled_for_emb['LABEL'].values
y_met_test = []
for lbl in test_labels_raw:
    if lbl in meta_label_encoder.classes_:
        y_met_test.append(meta_label_encoder.transform([lbl])[0])
    else:
        # Unseen label - assign to a default class (e.g., most common or first class)
        # This should be rare if data is properly split
        print(f"WARNING: Test label '{lbl}' not seen in training. Assigning to class 0.")
        y_met_test.append(0)  # Assign to first class as fallback
y_met_test = np.array(y_met_test)

if len(y_met_test) != len(test_data_scaled_for_emb):
    raise ValueError(
        f"ERROR: Label encoding mismatch. Expected {len(test_data_scaled_for_emb)} labels, got {len(y_met_test)}")

restricted_archs = [
    (128, 128, 128, 0.3, 1e-4, 'adam', 64, 20),
    (256, 64, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 64, 0.3, 1e-5, 'adam', 16, 20),
    (256, 256, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 256, 0.3, 1e-4, 'adam', 64, 20),
    (128, 128, 256, 0.3, 1e-5, 'adam', 32, 20),
    (256, 256, 128, 0.3, 1e-4, 'adam', 32, 20),
    (256, 128, 128, 0.3, 1e-5, 'adam', 64, 20)
]

# Add validation metrics to results columns
results_cols = ['Iteration', 'h1', 'h2', 'h3', 'dropout_rate', 'l2_reg', 'optimizer',
                'batch_size', 'epochs',
                'Val_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_F1',  # Validation metrics (for selection)
                'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_ROC_AUC',
                'Test_F1']  # Test metrics (for reporting)
results_cols += [f'Test_Accuracy_{lbl}' for lbl in met_class_labels]
results_df = pd.DataFrame(columns=results_cols)


def get_class_accuracies(y_true, y_pred, classes):
    out_ = []
    for i, cl_ in enumerate(classes):
        idx_ = np.where(y_true == i)[0]
        if len(idx_) > 0:
            out_.append(np.mean(y_pred[idx_] == i))
        else:
            out_.append(np.nan)
    return out_


###############################################################################
# 6) Train multiple meta-learner architectures on 'meta_feats_base_train'
#    Validate on 'meta_feats_meta_val'
###############################################################################
print("Evaluating restricted meta-learner architectures...")
print("Training on 80% of train+val, validating on 20% of train+val")
with timer("Meta-learner Architecture Selection"):
    # Safety check: Ensure meta-features are valid
    if meta_feats_base_train.shape[1] == 0:
        raise ValueError("ERROR: Meta-features have 0 dimensions! Cannot train meta-learner.")
    if len(y_base_train) != meta_feats_base_train.shape[0]:
        raise ValueError(
            f"ERROR: Label count mismatch. Expected {meta_feats_base_train.shape[0]} labels, got {len(y_base_train)}")

    for i, (h1, h2, h3, dr, l2r, opt, bsize, eps) in enumerate(restricted_archs, start=1):
        try:
            mm = create_meta_learner(h1=h1, h2=h2, h3=h3,
                                     dropout_rate=dr,
                                     l2_reg=l2r,
                                     optimizer=opt,
                                     input_dim=meta_feats_base_train.shape[1],
                                     num_classes=len(met_class_labels))
            # Train on base_train (80%), validate on meta_val (20% of train+val)
            history = mm.fit(meta_feats_base_train, y_base_train,
                             validation_data=(meta_feats_meta_val, y_meta_val),
                             epochs=eps, batch_size=bsize, verbose=1)

            # CRITICAL FIX: Evaluate on validation set for model selection (not test)
            y_pred_val = mm.predict(meta_feats_meta_val, verbose=0).argmax(axis=1)
            val_acc = accuracy_score(y_meta_val, y_pred_val)
            val_prec = precision_score(y_meta_val, y_pred_val, average='weighted', zero_division=0)
            val_rec = recall_score(y_meta_val, y_pred_val, average='weighted', zero_division=0)
            val_f1 = f1_score(y_meta_val, y_pred_val, average='weighted', zero_division=0)

        except Exception as e:
            print(f"WARNING: Failed to train meta-learner {i}: {str(e)}")
            print(f"  Skipping this architecture...")
            continue

        # Store each meta-learner model
        mm_path = os.path.join(evaluation_dir, f"meta_learner_{i}.h5")
        try:
            mm.save(mm_path)
        except Exception as e:
            print(f"WARNING: Failed to save meta-learner {i}: {str(e)}")
            continue

        # Evaluate on held-out test set => 'meta_feats_test' (for reporting only, not selection)
        try:
            y_pred_test = mm.predict(meta_feats_test, verbose=0).argmax(axis=1)
        except Exception as e:
            print(f"WARNING: Failed to predict with meta-learner {i}: {str(e)}")
            continue

        # Test metrics (for reporting, not selection)
        test_prec = precision_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
        test_rec = recall_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
        test_f1_ = f1_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
        test_acc = accuracy_score(y_met_test, y_pred_test)
        y_test_bin = to_categorical(y_met_test, num_classes=len(met_class_labels))
        y_pred_proba = mm.predict(meta_feats_test, verbose=0)
        try:
            test_roc_ = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            test_roc_ = np.nan

        cls_accs = get_class_accuracies(y_met_test, y_pred_test, met_class_labels)
        rowd = {
            'Iteration': i,
            'h1': h1, 'h2': h2, 'h3': h3,
            'dropout_rate': dr, 'l2_reg': l2r, 'optimizer': opt,
            'batch_size': bsize, 'epochs': eps,
            'Val_Accuracy': val_acc, 'Val_Precision': val_prec, 'Val_Recall': val_rec, 'Val_F1': val_f1,
            'Test_Accuracy': test_acc, 'Test_Precision': test_prec, 'Test_Recall': test_rec,
            'Test_ROC_AUC': test_roc_, 'Test_F1': test_f1_
        }
        for cacc_, clbl in zip(cls_accs, [f'Test_Accuracy_{x}' for x in met_class_labels]):
            rowd[clbl] = cacc_
        # Use pd.concat instead of deprecated append
        results_df = pd.concat([results_df, pd.DataFrame([rowd])], ignore_index=True)

# Save meta-learner results to CSV
meta_learner_csv_file = os.path.join(evaluation_dir, 'meta_learner_restricted_results.csv')
results_df.to_csv(meta_learner_csv_file, index=False)
print(f"Meta-learner results saved => {meta_learner_csv_file}")

###############################################################################
# 6a) Evaluate BASE CLASSIFIERS on held-out test set
###############################################################################
print("\n" + "=" * 80)
print("EVALUATING BASE CLASSIFIERS ON HELD-OUT TEST SET")
print("=" * 80)
base_classifier_accuracies = {}
for c_name, clf in met_classifiers.items():
    y_pred_base = clf.predict(combined_test)
    acc_base = accuracy_score(y_met_test, y_pred_base)
    base_classifier_accuracies[c_name] = acc_base
    print(f"  {c_name}: {acc_base:.4f}")

###############################################################################
# 6b) Evaluate BEST META-LEARNER on held-out test set
###############################################################################
results_df_sorted = results_df.sort_values(by='Val_Accuracy', ascending=False)  # Use Val_Accuracy for selection
best_model_row = results_df_sorted.iloc[0]
print(f"\nBest meta-learner selected based on VALIDATION accuracy: {best_model_row['Val_Accuracy']:.4f}")
print(f"  Test accuracy of selected model: {best_model_row['Test_Accuracy']:.4f}")
best_iteration = int(best_model_row['Iteration'])
h1 = best_model_row['h1']
h2 = best_model_row['h2']
h3 = best_model_row['h3']
dr = best_model_row['dropout_rate']
l2r = best_model_row['l2_reg']
opt = best_model_row['optimizer']
bsize = int(best_model_row['batch_size'])
eps = int(best_model_row['epochs'])

print(f"\nRe-training best meta-learner (iteration {best_iteration}) on 80% train+val...")
best_meta_model = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=meta_feats_base_train.shape[1],
    num_classes=len(met_class_labels)
)
best_meta_model.fit(meta_feats_base_train, y_base_train,
                    validation_data=(meta_feats_meta_val, y_meta_val),
                    epochs=eps, batch_size=bsize, verbose=0)

# Evaluate on held-out test set
y_pred_meta_test = best_meta_model.predict(meta_feats_test).argmax(axis=1)
meta_accuracy = accuracy_score(y_met_test, y_pred_meta_test)
base_classifier_accuracies['Meta'] = meta_accuracy
print(f"  Meta-learner (selected by Val Acc={best_model_row['Val_Accuracy']:.4f}): {meta_accuracy:.4f}")
print("=" * 80 + "\n")

###############################################################################
# 6c) Create bar plot: Classifier Accuracy on Test Split
###############################################################################
print("Creating classifier accuracy comparison plot...")
classifier_names = ['LogisticRegression', 'KNN', 'RandomForest', 'SVM', 'XGBoost', 'Meta']
accuracies = [base_classifier_accuracies.get(name, 0.0) for name in classifier_names]

# Safety check: Ensure we have accuracies for all classifiers
if len(accuracies) != len(classifier_names):
    raise ValueError(
        f"ERROR: Accuracy count mismatch. Expected {len(classifier_names)} accuracies, got {len(accuracies)}")

# Create plot exactly like the image
fig, ax = plt.subplots(figsize=(10, 6))

# Colors: black for base classifiers, red for Meta
colors = ['black' if name != 'Meta' else 'red' for name in classifier_names]
bars = ax.bar(classifier_names, accuracies, color=colors, width=0.6)

# Add horizontal reference line (dotted grey) at mean of base classifiers
base_mean = np.mean([acc for name, acc in zip(classifier_names, accuracies) if name != 'Meta'])
ax.axhline(y=base_mean, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)

# Formatting - EXACTLY like the image
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Classifier', fontsize=12)
ax.set_title('Classifier Accuracy on Test Split', fontsize=14, fontweight='bold')
ax.set_ylim([0.80, 1.00])
ax.set_yticks(np.arange(0.80, 1.01, 0.025))
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

# Rotate x-axis labels (diagonal)
plt.xticks(rotation=45, ha='right')

# No value labels on bars (image doesn't show them)

plt.tight_layout()
plot_path = os.path.join(evaluation_dir, 'classifier_accuracy_test_split.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Classifier accuracy plot saved => {plot_path}")

###############################################################################
# 7) Evaluate the top-2 meta-learners on subsets (TCGA+GTEX vs. MET)
#    using the same SNN+CAE embeddings => base classifiers => meta-features
###############################################################################
print("Preparing subsets for Confusion Matrices of the best meta-learners...")

# A) Subset test_data_scaled_for_emb
tcga_gtex_test = test_data_scaled_for_emb[test_data_scaled_for_emb['SOURCE'].isin(['TCGA', 'GTEX'])].copy()
met_test = test_data_scaled_for_emb[test_data_scaled_for_emb['SOURCE'] == 'METASTATIC'].copy()

# B) True labels (numeric) - handle unseen labels gracefully
y_true_tcga_gtex = []
for lbl in tcga_gtex_test['LABEL'].values:
    if lbl in meta_label_encoder.classes_:
        y_true_tcga_gtex.append(meta_label_encoder.transform([lbl])[0])
    else:
        print(f"WARNING: TCGA/GTEx test label '{lbl}' not seen in training. Assigning to class 0.")
        y_true_tcga_gtex.append(0)
y_true_tcga_gtex = np.array(y_true_tcga_gtex)

y_true_met = []
for lbl in met_test['LABEL'].values:
    if lbl in meta_label_encoder.classes_:
        y_true_met.append(meta_label_encoder.transform([lbl])[0])
    else:
        print(f"WARNING: Metastatic test label '{lbl}' not seen in training. Assigning to class 0.")
        y_true_met.append(0)
y_true_met = np.array(y_true_met)

# C) Generate EMBEDDINGS for these subsets => shape=(None,128)
X_tg_test_emb = np.concatenate([
    snn_base_network.predict(tcga_gtex_test[fcols].values),
    cae_encoder.predict(tcga_gtex_test[fcols].values)
], axis=1)
X_met_test_emb = np.concatenate([
    snn_base_network.predict(met_test[fcols].values),
    cae_encoder.predict(met_test[fcols].values)
], axis=1)

# D) Convert embeddings -> meta-features by feeding each embedding to base classifiers
meta_feats_tg_test = get_meta_features(met_classifiers, X_tg_test_emb)
meta_feats_met_test = get_meta_features(met_classifiers, X_met_test_emb)

# E) Sort and pick top-2 by VALIDATION accuracy (not test - prevents overfitting)
results_df_sorted = results_df.sort_values(by='Val_Accuracy', ascending=False)
best_two_models = results_df_sorted.head(2).copy()
print(f"Top-2 meta-learners selected based on VALIDATION accuracy")

tg_classes = meta_label_encoder.classes_
met_classes = meta_label_encoder.classes_

# Use the already-trained best_meta_model for the first one, retrain second if needed
for idx, (model_idx, row) in enumerate(best_two_models.iterrows()):
    iteration = int(row['Iteration'])
    h1 = row['h1']
    h2 = row['h2']
    h3 = row['h3']
    dr = row['dropout_rate']
    l2r = row['l2_reg']
    opt = row['optimizer']
    bsize = int(row['batch_size'])
    eps = int(row['epochs'])

    if idx == 0:
        # Use the already-trained best model
        print(f"\nUsing best meta-learner from iteration={iteration} with Test_Accuracy={row['Test_Accuracy']:.3f}")
        meta_model = best_meta_model
    else:
        # Retrain the second best model
        print(
            f"\nRe-initializing meta-learner from iteration={iteration} with Test_Accuracy={row['Test_Accuracy']:.3f}")
        meta_model = create_meta_learner(
            h1=h1, h2=h2, h3=h3,
            dropout_rate=dr, l2_reg=l2r, optimizer=opt,
            input_dim=meta_feats_base_train.shape[1],
            num_classes=len(met_class_labels)
        )
        meta_model.fit(meta_feats_base_train, y_base_train,
                       validation_data=(meta_feats_meta_val, y_meta_val),
                       epochs=eps, batch_size=bsize, verbose=0)

    # Also store these top-2 meta models
    best_path = os.path.join(evaluation_dir, f"best_meta_learner_{iteration}.h5")
    meta_model.save(best_path)

    # G) Predict integer labels on meta-features of each subset
    y_pred_tcga_gtex = meta_model.predict(meta_feats_tg_test).argmax(axis=1)
    y_pred_met = meta_model.predict(meta_feats_met_test).argmax(axis=1)

    # ------------------------------------------------------------------------
    # Convert BOTH true & predicted labels from int -> string for confusion_matrix
    # ------------------------------------------------------------------------
    y_true_labels_tg_str = [tg_classes[i] for i in y_true_tcga_gtex]
    y_true_labels_met_str = [met_classes[i] for i in y_true_met]

    y_pred_labels_tg_str = [tg_classes[i] for i in y_pred_tcga_gtex]
    y_pred_labels_met_str = [met_classes[i] for i in y_pred_met]

    # 1) TCGA+GTEX confusion matrices
    # Filter out metastatic labels from tg_classes
    non_meta_classes = [c for c in tg_classes if '_METASTATIC' not in c]

    # -- (a) "Absolute" numbers in cells, but color by fraction so 30/30 and 600/600
    #    get the same color on the diagonal.
    plot_confusion_matrix(
        y_true_labels_tg_str,
        y_pred_labels_tg_str,
        non_meta_classes,  # filtered classes
        f"TCGA-GTEX Confusion Matrix (Absolute) - Model_{iteration}",
        os.path.join(evaluation_dir, f"TCGA_GTEX_CM_abs_model_{iteration}.png"),
        normalize=True,  # color by row fraction
        no_numbers=False
    )

    # -- (b) Normalized matrix (both color and numbers are fraction)
    plot_confusion_matrix(
        y_true_labels_tg_str,
        y_pred_labels_tg_str,
        non_meta_classes,
        f"TCGA-GTEX Confusion Matrix (Normalized) - Model_{iteration}",
        os.path.join(evaluation_dir, f"TCGA_GTEX_CM_norm_model_{iteration}.png"),
        normalize=True,
        no_numbers=False
    )

    # -- (c) Normalized with no numbers shown
    plot_confusion_matrix(
        y_true_labels_tg_str,
        y_pred_labels_tg_str,
        non_meta_classes,
        f"TCGA-GTEX Confusion Matrix (Normalized No Numbers) - Model_{iteration}",
        os.path.join(evaluation_dir, f"TCGA_GTEX_CM_norm_nonum_model_{iteration}.png"),
        normalize=True,
        no_numbers=True
    )

    # 2) Metastatic confusion matrices, but ONLY the labels truly present
    all_labels_used = set(y_true_labels_met_str) | set(y_pred_labels_met_str)
    actual_meta_labels = sorted(list(all_labels_used))  # only what's truly present

    plot_confusion_matrix(
        y_true_labels_met_str,
        y_pred_labels_met_str,
        actual_meta_labels,
        f"Metastatic Confusion Matrix (Absolute) - Model_{iteration}",
        os.path.join(evaluation_dir, f"Metastatic_CM_abs_model_{iteration}.png"),
        normalize=False,
        no_numbers=False
    )

    plot_confusion_matrix(
        y_true_labels_met_str,
        y_pred_labels_met_str,
        actual_meta_labels,
        f"Metastatic Confusion Matrix (Normalized) - Model_{iteration}",
        os.path.join(evaluation_dir, f"Metastatic_CM_norm_model_{iteration}.png"),
        normalize=True,
        no_numbers=False
    )

    plot_confusion_matrix(
        y_true_labels_met_str,
        y_pred_labels_met_str,
        actual_meta_labels,
        f"Metastatic Confusion Matrix (Normalized No Numbers) - Model_{iteration}",
        os.path.join(evaluation_dir, f"Metastatic_CM_norm_nonum_model_{iteration}.png"),
        normalize=True,
        no_numbers=True
    )
# --- New snippet: Confusion Matrices for Non-Metastatic Tissues (Counts) ---
from sklearn.metrics import confusion_matrix

# Get all non-met labels 
non_met_classes = sorted(list(set(y_true_labels_tg_str)))

# Compute the full confusion matrix (absolute counts) for non-met tissues
cm_full = confusion_matrix(y_true_labels_tg_str, y_pred_labels_tg_str, labels=non_met_classes)

# Plot the full confusion matrix (counts) with blue color coding and font size 14
plt.figure(figsize=(max(8, len(non_met_classes) * 0.8), max(6, len(non_met_classes) * 0.6)))
sns.heatmap(cm_full,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            annot_kws={'fontsize': 14},
            xticklabels=non_met_classes,
            yticklabels=non_met_classes)
plt.title(f"Non-Metastatic Confusion Matrix (Counts) - Model_{iteration}", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(evaluation_dir, f"NonMet_CM_abs_full_model_{iteration}.png"), dpi=300)
plt.close()

# Now create sliced confusion matrices (blocks of at most 10x10 labels)
block_size = 11
num_labels = len(non_met_classes)
block_index = 0
for start in range(0, num_labels, block_size):
    end = min(start + block_size, num_labels)
    # Slice both rows and columns from the full confusion matrix
    cm_slice = cm_full[start:end, start:end]
    classes_slice = non_met_classes[start:end]

    plt.figure(figsize=(max(8, (end - start) * 0.8), max(6, (end - start) * 0.6)))
    sns.heatmap(cm_slice,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=True,
                annot_kws={'fontsize': 14},
                xticklabels=classes_slice,
                yticklabels=classes_slice)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_dir, f"NonMet_CM_abs_slice_{block_index + 1}_model_{iteration}.png"), dpi=300)
    plt.close()
    block_index += 1
# -------------- Plot 2D & 3D Dim Reductions --------------
print("Plotting dimensionality reductions for test data vs. entire reference set...")


met_only = test_data_scaled_for_emb[test_data_scaled_for_emb['SOURCE'] == 'METASTATIC']
tcga_gtex = test_data_scaled[test_data_scaled['SOURCE'].isin(['TCGA', 'GTEX'])]

X_met_pre = met_only[fcols].values
X_met_post_snn = snn_base_network.predict(met_only[fcols].values)
X_met_post_cae = cae_encoder.predict(met_only[fcols].values)
X_met_post = np.concatenate([X_met_post_snn, X_met_post_cae], axis=1)
met_labels = met_only['LABEL'].values
unique_met_labels = np.unique(met_labels)

X_tgc_pre = tcga_gtex[fcols].values
X_tgc_post_snn = snn_base_network.predict(tcga_gtex[fcols].values)
X_tgc_post_cae = cae_encoder.predict(tcga_gtex[fcols].values)
X_tgc_post = np.concatenate([X_tgc_post_snn, X_tgc_post_cae], axis=1)
tgc_labels = tcga_gtex['LABEL'].values
unique_tgc_labels = np.unique(tgc_labels)


def plot_dim_reductions_2d(X_pre, X_post, labels, set_name, output_dir, unique_labels, hide_legend=False):
    """
    Creates a 2D dimension reduction figure using t-SNE, PCA, and UMAP.
    In each subplot the top-right annotation shows only the median distortion.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from umap import UMAP
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Compute 2D projections for the "pre" data
    tsne_pre = TSNE(n_components=2, random_state=42).fit_transform(X_pre)
    pca = PCA(n_components=2, random_state=42)
    pca_pre = pca.fit_transform(X_pre)
    umap_model = UMAP(n_components=2, random_state=42)
    umap_pre = umap_model.fit_transform(X_pre)

    # Compute 2D projections for the "post" data
    tsne_post = TSNE(n_components=2, random_state=42).fit_transform(X_post)
    pca_post = pca.fit_transform(X_post)
    umap_post = umap_model.fit_transform(X_post)

    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    def scatter_2d(ax, coords, lbls, title):
        # Plot points for each label
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=[color_map[lbl]], label=str(lbl), s=20, alpha=0.6)
        ax.set_title(title)
        # Compute median distortion across clusters:
        all_distortions = []
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            if len(idx) == 0:
                continue
            centroid = coords[idx].mean(axis=0)
            dists = np.linalg.norm(coords[idx] - centroid, axis=1)
            all_distortions.append(np.mean(dists))
        if all_distortions:
            median_dist = np.median(all_distortions)
            ax.text(0.95, 0.95, f"Median Dist={median_dist:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    scatter_2d(axes[0], tsne_pre, np.array(labels), "t-SNE (Pre)")
    scatter_2d(axes[1], pca_pre, np.array(labels), "PCA (Pre)")
    scatter_2d(axes[2], umap_pre, np.array(labels), "UMAP (Pre)")
    scatter_2d(axes[3], tsne_post, np.array(labels), "t-SNE (Post)")
    scatter_2d(axes[4], pca_post, np.array(labels), "PCA (Post)")
    scatter_2d(axes[5], umap_post, np.array(labels), "UMAP (Post)")

    if not hide_legend:
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   ncol=4, fontsize='small')

    fig.suptitle(f"{set_name} 2D Dimensionality Reduction", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"{set_name}_2D_dim_reduction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dim_reductions_3d(X_pre, X_post, labels, set_name, output_dir, unique_labels, hide_legend=False):
    """
    Create 3D dimensionality reduction visualization comparing pre- and post-embedding data.

    Generates side-by-side comparisons of 3D t-SNE, PCA, and UMAP projections for data
    before and after embedding transformation. Each subplot displays median distortion
    metric to quantify cluster compactness in 3D space.

    Args:
        X_pre (np.array): High-dimensional data before embedding (n_samples, n_features)
        X_post (np.array): Low-dimensional embeddings after transformation (n_samples, n_embedding)
        labels (np.array): Tissue labels for each sample (n_samples,)
        set_name (str): Name identifier for the dataset (e.g., "Train", "Test")
        output_dir (str): Directory to save the figure
        unique_labels (list): List of unique tissue labels for color mapping
        hide_legend (bool): If True, skip legend display. Default: False

    Returns:
        None: Saves figure as '{set_name}_3D_dim_reduction.png' in output_dir

    Notes:
        - Creates 2x3 subplot grid: top row=pre-embedding, bottom row=post-embedding
        - Columns: t-SNE, PCA, UMAP (all in 3D)
        - Median distortion computed as mean distance from cluster centroid in 3D
        - Uses fixed random_state=42 for reproducibility
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from umap import UMAP
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    fig = plt.figure(figsize=(15, 9))

    # Compute 3D projections for the "pre" data
    tsne_pre = TSNE(n_components=3, random_state=42).fit_transform(X_pre)
    pca = PCA(n_components=3, random_state=42)
    pca_pre = pca.fit_transform(X_pre)
    umap_model = UMAP(n_components=3, random_state=42)
    umap_pre = umap_model.fit_transform(X_pre)

    # Compute 3D projections for the "post" data
    tsne_post = TSNE(n_components=3, random_state=42).fit_transform(X_post)
    pca_post = pca.fit_transform(X_post)
    umap_post = umap_model.fit_transform(X_post)

    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    def scatter_3d(ax, coords, lbls, title):
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                       c=[color_map[lbl]], label=str(lbl), s=20, alpha=0.6)
        ax.set_title(title)
        # Compute median distortion for the subplot
        all_distortions = []
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            if len(idx) == 0:
                continue
            centroid = coords[idx].mean(axis=0)
            dists = np.linalg.norm(coords[idx] - centroid, axis=1)
            all_distortions.append(np.mean(dists))
        if all_distortions:
            median_dist = np.median(all_distortions)
            ax.text2D(0.95, 0.95, f"Median Dist={median_dist:.2f}",
                      transform=ax.transAxes, fontsize=9,
                      ha='right', va='top',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # Create a 2x3 grid of subplots: top row = pre, bottom row = post for each method.
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')

    scatter_3d(ax1, tsne_pre, np.array(labels), "t-SNE (Pre)")
    scatter_3d(ax2, pca_pre, np.array(labels), "PCA (Pre)")
    scatter_3d(ax3, umap_pre, np.array(labels), "UMAP (Pre)")
    scatter_3d(ax4, tsne_post, np.array(labels), "t-SNE (Post)")
    scatter_3d(ax5, pca_post, np.array(labels), "PCA (Post)")
    scatter_3d(ax6, umap_post, np.array(labels), "UMAP (Post)")

    handles, lbls = ax1.get_legend_handles_labels()
    if not hide_legend:
        fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   ncol=4, fontsize='small')

    fig.suptitle(f"{set_name} 3D Dimensionality Reduction", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"{set_name}_3D_dim_reduction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Call the modified 2D and 3D plotting functions:
plot_dim_reductions_2d(X_met_pre, X_met_post, met_labels,
                       "MetastaticOnly", evaluation_dir, unique_met_labels,
                       hide_legend=False)
plot_dim_reductions_2d(X_tgc_pre, X_tgc_post, tgc_labels,
                       "TCGA_GTEX", evaluation_dir, unique_tgc_labels,
                       hide_legend=True)

plot_dim_reductions_3d(X_met_pre, X_met_post, met_labels,
                       "MetastaticOnly", evaluation_dir, unique_met_labels,
                       hide_legend=False)
plot_dim_reductions_3d(X_tgc_pre, X_tgc_post, tgc_labels,
                       "TCGA_GTEX", evaluation_dir, unique_tgc_labels,
                       hide_legend=True)

###############################################################################
# 3D PCA & 3D tSNE 4-Axis Plots (Non-Met vs. Met, Pre vs. Post), with Cluster Centroids
###############################################################################
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from sklearn.metrics import silhouette_samples  # (Not used anymore for display)

def _scatter_3d_with_clusters(ax, coords, labels, unique_labels, title=""):
    """
    Plots points in 3D, colored by label, plus:
      - a hollow circle at each label’s centroid,
      - a text label (next to the centroid) showing only the distortion,
      - and a top‐right annotation (in axis coordinates) showing the median distortion.
    """
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {ul: palette[i] for i, ul in enumerate(unique_labels)}

    # Scatter points for each label
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1], coords[idx, 2],
            c=[color_map[lbl]],
            label=str(lbl),
            s=20, alpha=0.6
        )

    # Compute centroids and distortion for each label
    all_distortions = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        centroid = coords[idx].mean(axis=0)
        dists = np.linalg.norm(coords[idx] - centroid, axis=1)
        distortion_lbl = np.mean(dists)
        all_distortions.append(distortion_lbl)

        # Plot a hollow circle at the centroid
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            s=120,
            facecolors='none',
            edgecolors=[color_map[lbl]],
            linewidth=2,
            marker='o'
        )

        # Place text near the centroid showing only the distortion value
        offset = 0.5
        ax.text(
            centroid[0] + offset,
            centroid[1] + offset,
            centroid[2],
            f"({distortion_lbl:.2f})",
            color=color_map[lbl],
            fontsize=8
        )

    # Compute and show the median distortion in the top-right corner of the axis
    if all_distortions:
        median_dist = np.median(all_distortions)
        ax.text2D(
            0.95, 0.95,
            f"Median Dist={median_dist:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        )
    ax.set_title(title)


def plot_4_axes_3d(
        X_pre_nonmet, X_pre_met, X_post_nonmet, X_post_met,
        labels_pre_nonmet, labels_pre_met, labels_post_nonmet, labels_post_met,
        unique_labels,
        method='pca',
        fig_title='',
        out_png='4axes_3d.png'
):
    """
    Creates a 2x2 figure with each subplot a 3D projection:
      - [row0, col0]: Non-Met PRE
      - [row0, col1]: Non-Met POST
      - [row1, col0]: Met PRE
      - [row1, col1]: Met POST
    Each subplot uses the specified dimension-reduction method (PCA or TSNE)
    and displays points, cluster centroids, and the median distortion in the top right.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import os

    fig = plt.figure(figsize=(14, 10))

    ax00 = fig.add_subplot(2, 2, 1, projection='3d')
    ax01 = fig.add_subplot(2, 2, 2, projection='3d')
    ax10 = fig.add_subplot(2, 2, 3, projection='3d')
    ax11 = fig.add_subplot(2, 2, 4, projection='3d')

    def transform_3d(X):
        if method.lower() == 'pca':
            pca = PCA(n_components=3, random_state=42)
            return pca.fit_transform(X)
        else:
            tsne_3d = TSNE(n_components=3, random_state=42)
            return tsne_3d.fit_transform(X)

    coords_pre_nonmet = transform_3d(X_pre_nonmet)
    coords_pre_met = transform_3d(X_pre_met)
    coords_post_nonmet = transform_3d(X_post_nonmet)
    coords_post_met = transform_3d(X_post_met)

    _scatter_3d_with_clusters(
        ax00, coords_pre_nonmet, labels_pre_nonmet, unique_labels,
        title="Non-Met PRE"
    )
    _scatter_3d_with_clusters(
        ax01, coords_post_nonmet, labels_post_nonmet, unique_labels,
        title="Non-Met POST"
    )
    _scatter_3d_with_clusters(
        ax10, coords_pre_met, labels_pre_met, unique_labels,
        title="Met PRE"
    )
    _scatter_3d_with_clusters(
        ax11, coords_post_met, labels_post_met, unique_labels,
        title="Met POST"
    )

    handles, lbls = ax00.get_legend_handles_labels()
    fig.legend(
        handles, lbls,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize='small',
        frameon=True
    )

    fig.suptitle(f"{fig_title} (3D {method.upper()})", y=0.95, fontsize=14)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


print("\n--- Plotting 3D PCA & 3D tSNE (4-axis) for each embedding type ---")

# Separate Non-Met vs. Met in the test set
test_nonmet_df = test_data_scaled[test_data_scaled['SOURCE'] != 'METASTATIC'].copy()
test_met_df = test_data_scaled[test_data_scaled['SOURCE'] == 'METASTATIC'].copy()

# 'Pre' = scaled gene expression
X_pre_nonmet = test_nonmet_df[fcols].values
X_pre_met = test_met_df[fcols].values

labels_pre_nonmet = test_nonmet_df['LABEL'].values
labels_pre_met = test_met_df['LABEL'].values

# SNN embeddings alone
X_snn_nonmet = snn_base_network.predict(X_pre_nonmet)
X_snn_met = snn_base_network.predict(X_pre_met)

# CAE embeddings alone
X_cae_nonmet = cae_encoder.predict(X_pre_nonmet)
X_cae_met = cae_encoder.predict(X_pre_met)

# Combined embeddings (SNN + CAE)
X_combined_nonmet = np.concatenate([X_snn_nonmet, X_cae_nonmet], axis=1)
X_combined_met = np.concatenate([X_snn_met, X_cae_met], axis=1)

unique_labels_plot = np.unique(np.concatenate([labels_pre_nonmet, labels_pre_met]))

# Create 6 Figures: (Combined, SNN, CAE) x (PCA, TSNE)

# (A) Combined
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_combined_nonmet, X_combined_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='pca',
    fig_title="Combined Embeddings",
    out_png=os.path.join(evaluation_dir, "combined_3dPCA_4axes.png")
)
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_combined_nonmet, X_combined_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='tsne',
    fig_title="Combined Embeddings",
    out_png=os.path.join(evaluation_dir, "combined_3dTSNE_4axes.png")
)

# (B) SNN
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_snn_nonmet, X_snn_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='pca',
    fig_title="SNN Embeddings",
    out_png=os.path.join(evaluation_dir, "snn_3dPCA_4axes.png")
)
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_snn_nonmet, X_snn_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='tsne',
    fig_title="SNN Embeddings",
    out_png=os.path.join(evaluation_dir, "snn_3dTSNE_4axes.png")
)

# (C) CAE
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_cae_nonmet, X_cae_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='pca',
    fig_title="CAE Embeddings",
    out_png=os.path.join(evaluation_dir, "cae_3dPCA_4axes.png")
)
plot_4_axes_3d(
    X_pre_nonmet, X_pre_met, X_cae_nonmet, X_cae_met,
    labels_pre_nonmet, labels_pre_met, labels_pre_nonmet, labels_pre_met,
    unique_labels=unique_labels_plot,
    method='tsne',
    fig_title="CAE Embeddings",
    out_png=os.path.join(evaluation_dir, "cae_3dTSNE_4axes.png")
)

print("Finished generating 3D PCA & 3D tSNE plots (non-met vs met, pre vs post).")


###############################################################################
# Plot ALL Samples (including metastatic tissue) using PCA (2D and 3D) – Only Distortion
###############################################################################
def plot_all_samples_pca(test_df,
                         feature_cols,
                         snn_net,
                         cae_enc,
                         output_path,
                         legend_ncol: int = 8):
    """
    2×2 PCA summary of ALL samples   (Pre vs Post, 2-D & 3-D)

    • colour = LABEL   • median distortion printed on each panel
    • big legend under the figure (font tracks rcParams)
    """
    # ---------- imports (FIRST!) ----------
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.decomposition import PCA
    import matplotlib.patches as mpatches

    # ---------- embeddings ----------
    X_all_pre = test_df[feature_cols].values
    X_all_post = np.concatenate(
        [
            snn_net.predict(X_all_pre, verbose=0),
            cae_enc.predict(X_all_pre, verbose=0)
        ],
        axis=1,
    )

    all_labels = test_df["LABEL"].values
    unique_labels = np.unique(all_labels)

    # consistent colours
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # ---------- helpers ----------
    def _median_dist(pts, lbls):
        vals = []
        for ul in unique_labels:
            idx = np.where(lbls == ul)[0]
            if idx.size:
                centre = pts[idx].mean(axis=0)
                vals.append(np.mean(np.linalg.norm(pts[idx] - centre, axis=1)))
        return float(np.median(vals)) if vals else np.nan

    def _scatter_2d(ax, pts, lbls, title):
        for ul in unique_labels:
            idx = np.where(lbls == ul)[0]
            ax.scatter(pts[idx, 0], pts[idx, 1],
                       s=20, color=[color_map[ul]], alpha=0.6)
        ax.set_title(title)
        md = _median_dist(pts, lbls)
        ax.text(0.97, 0.97, f"Median Dist={md:.2f}",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=mpl.rcParams["axes.labelsize"],
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec="black", alpha=0.7))

    def _scatter_3d(ax, pts, lbls, title):
        for ul in unique_labels:
            idx = np.where(lbls == ul)[0]
            ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                       s=20, color=[color_map[ul]], alpha=0.6)
        ax.set_title(title)
        md = _median_dist(pts, lbls)
        ax.text2D(0.97, 0.97, f"Median Dist={md:.2f}",
                  transform=ax.transAxes,
                  ha="right", va="top",
                  fontsize=mpl.rcParams["axes.labelsize"],
                  bbox=dict(boxstyle="round,pad=0.3",
                            fc="white", ec="black", alpha=0.7))

    # ---------- PCA projections ----------
    pca2 = PCA(n_components=2, random_state=42)
    pre_2d = pca2.fit_transform(X_all_pre)
    post_2d = pca2.fit_transform(X_all_post)

    pca3 = PCA(n_components=3, random_state=42)
    pre_3d = pca3.fit_transform(X_all_pre)
    post_3d = pca3.fit_transform(X_all_post)

    # ---------- figure ----------
    fig = plt.figure(figsize=(12, 10))

    # 2-D
    ax00 = plt.subplot2grid((2, 2), (0, 0))
    ax01 = plt.subplot2grid((2, 2), (0, 1))
    _scatter_2d(ax00, pre_2d, all_labels, "ALL Samples Pre (2D PCA)")
    _scatter_2d(ax01, post_2d, all_labels, "ALL Samples Post (2D PCA)")

    # 3-D
    ax10 = plt.subplot2grid((2, 2), (1, 0), projection="3d")
    ax11 = plt.subplot2grid((2, 2), (1, 1), projection="3d")
    _scatter_3d(ax10, pre_3d, all_labels, "ALL Samples Pre (3D PCA)")
    _scatter_3d(ax11, post_3d, all_labels, "ALL Samples Post (3D PCA)")

    fig.suptitle("PCA of ALL Samples (Pre vs. Post)",
                 fontsize=mpl.rcParams["figure.titlesize"])
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


###############################################################################
# Plot ALL Samples (including metastatic tissue) using PCA (2D and 3D) – Distortion & Silhouette
###############################################################################
def plot_all_samples_pca_with_sil(test_df, feature_cols, snn_net, cae_enc, output_path):
    """
    Plots a 2x2 figure of ALL samples (including metastatic) as follows:
      - ax0: 2D PCA on pre (raw scaled gene expression)
      - ax1: 2D PCA on post (combined SNN+CAE embeddings)
      - ax2: 3D PCA on pre
      - ax3: 3D PCA on post
    In every subplot the top-right displays both the median distortion and the median silhouette.
    No legend is shown.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_samples
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Pre data: raw scaled gene expression for all samples
    X_all_pre = test_df[feature_cols].values
    # Post data: combined embeddings from SNN and CAE
    X_all_post_snn = snn_net.predict(X_all_pre)
    X_all_post_cae = cae_enc.predict(X_all_pre)
    X_all_post = np.concatenate([X_all_post_snn, X_all_post_cae], axis=1)

    # For coloring, get the labels
    all_labels = test_df['LABEL'].values
    unique_labels = np.unique(all_labels)

    # Prepare a colormap
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # For silhouette, we need numeric labels:
    label_to_num = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[lbl] for lbl in all_labels])

    # Compute silhouette values for the entire dataset (if >1 cluster)
    if len(unique_labels) > 1:
        sil_values = silhouette_samples(X_all_pre, numeric_labels)
    else:
        sil_values = np.zeros(len(X_all_pre))

    def scatter_2d(ax, X, labels, sil_vals):
        # Plot points for each label
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            ax.scatter(X[idx, 0], X[idx, 1],
                       c=[color_map[lbl]], s=20, alpha=0.6)
        # For each label, compute median distortion and median silhouette
        dist_list = []
        sil_list = []
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue
            centroid = X[idx].mean(axis=0)
            dists = np.linalg.norm(X[idx] - centroid, axis=1)
            dist_list.append(np.mean(dists))
            # For silhouette, use precomputed sil_vals corresponding to these indices
            sil_list.append(np.median(sil_vals[idx]))
        if dist_list and sil_list:
            median_dist = np.median(dist_list)
            median_sil = np.median(sil_list)
            ax.text(0.95, 0.95, f"Median Dist={median_dist:.2f}\nMedian Sil={median_sil:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    def scatter_3d(ax, X, labels, sil_vals):
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2],
                       c=[color_map[lbl]], s=20, alpha=0.6)
        dist_list = []
        sil_list = []
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) == 0:
                continue
            centroid = X[idx].mean(axis=0)
            dists = np.linalg.norm(X[idx] - centroid, axis=1)
            dist_list.append(np.mean(dists))
            sil_list.append(np.median(sil_vals[idx]))
        if dist_list and sil_list:
            median_dist = np.median(dist_list)
            median_sil = np.median(sil_list)
            ax.text2D(0.95, 0.95, f"Median Dist={median_dist:.2f}\nMedian Sil={median_sil:.2f}",
                      transform=ax.transAxes, fontsize=9,
                      ha='right', va='top',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # Compute 2D PCA for pre and post data
    pca_2d = PCA(n_components=2, random_state=42)
    X_all_pre_2d = pca_2d.fit_transform(X_all_pre)
    X_all_post_2d = pca_2d.fit_transform(X_all_post)

    # Compute 3D PCA for pre and post data
    pca_3d = PCA(n_components=3, random_state=42)
    X_all_pre_3d = pca_3d.fit_transform(X_all_pre)
    X_all_post_3d = pca_3d.fit_transform(X_all_post)

    # Create a 2x2 figure: ax0 and ax1 for 2D; ax2 and ax3 for 3D.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    scatter_2d(axs[0, 0], X_all_pre_2d, all_labels, sil_values)
    axs[0, 0].set_title("ALL Samples Pre (2D PCA)")
    scatter_2d(axs[0, 1], X_all_post_2d, all_labels, sil_values)
    axs[0, 1].set_title("ALL Samples Post (2D PCA)")

    # For 3D subplots, create them with projection='3d'
    ax2 = plt.subplot(2, 2, 3, projection='3d')
    ax3 = plt.subplot(2, 2, 4, projection='3d')
    scatter_3d(ax2, X_all_pre_3d, all_labels, sil_values)
    ax2.set_title("ALL Samples Pre (3D PCA)")
    scatter_3d(ax3, X_all_post_3d, all_labels, sil_values)
    ax3.set_title("ALL Samples Post (3D PCA)")

    # Do not add a legend in this figure.
    fig.suptitle("PCA of ALL Samples (Pre vs. Post) with Silhouette", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Call the new plot functions for all samples:
all_samples_output_path = os.path.join(evaluation_dir, "all_samples_PCA_pre_post.png")
plot_all_samples_pca(test_data_scaled, fcols, snn_base_network, cae_encoder, all_samples_output_path)
print("Finished generating ALL samples PCA plot (2D and 3D, only distortion).")

all_samples_with_sil_output_path = os.path.join(evaluation_dir, "all_samples_PCA_pre_post_with_sil.png")
plot_all_samples_pca_with_sil(test_data_scaled, fcols, snn_base_network, cae_encoder, all_samples_with_sil_output_path)
print("Finished generating ALL samples PCA plot (2D and 3D, with distortion and silhouette).")

###############################################################################
# Compare SNN-only vs CAE-only vs Combined Meta-Learners
# Blue=Accuracy, Red=F1
###############################################################################
print("\n--- Comparing SNN, CAE, and Combined embeddings via meta-learner ---")

# 1) Retrieve "best" meta-learner architecture from earlier
best_model_row = best_two_models.iloc[0]  # pick the single best row
h1 = int(best_model_row['h1'])
h2 = int(best_model_row['h2'])
h3 = int(best_model_row['h3'])
dr = float(best_model_row['dropout_rate'])
l2r = float(best_model_row['l2_reg'])
opt = best_model_row['optimizer']
bsize = int(best_model_row['batch_size'])
eps = int(best_model_row['epochs'])

# 2) We'll train a meta-learner on SNN embeddings only (using new nested splits)
print("Training meta-learner on SNN embeddings alone...")
# Get SNN embeddings for the nested splits
snn_emb_base_train = snn_base_network.predict(train_val_base[fcols].values)
snn_emb_meta_val = snn_base_network.predict(train_val_meta[fcols].values)

mm_snn = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=snn_emb_base_train.shape[1],  # only SNN dims
    num_classes=len(met_class_labels)
)
mm_snn.fit(snn_emb_base_train, y_base_train,
           validation_data=(snn_emb_meta_val, y_meta_val),
           epochs=eps, batch_size=bsize, verbose=0)
y_pred_snn_test = mm_snn.predict(snn_emb_test).argmax(axis=1)

acc_snn = accuracy_score(y_met_test, y_pred_snn_test)
f1_snn = f1_score(y_met_test, y_pred_snn_test, average='weighted', zero_division=0)

print(f"SNN-Only => Accuracy={acc_snn:.3f}, F1={f1_snn:.3f}")

# 3) Train a meta-learner on CAE embeddings only (using new nested splits)
print("Training meta-learner on CAE embeddings alone...")
# Get CAE embeddings for the nested splits
cae_emb_base_train = cae_encoder.predict(train_val_base[fcols].values)
cae_emb_meta_val = cae_encoder.predict(train_val_meta[fcols].values)

mm_cae = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=cae_emb_base_train.shape[1],  # only CAE dims
    num_classes=len(met_class_labels)
)
mm_cae.fit(cae_emb_base_train, y_base_train,
           validation_data=(cae_emb_meta_val, y_meta_val),
           epochs=eps, batch_size=bsize, verbose=0)
y_pred_cae_test = mm_cae.predict(cae_emb_test).argmax(axis=1)

acc_cae = accuracy_score(y_met_test, y_pred_cae_test)
f1_cae = f1_score(y_met_test, y_pred_cae_test, average='weighted', zero_division=0)

print(f"CAE-Only => Accuracy={acc_cae:.3f}, F1={f1_cae:.3f}")

# 4) "Combined" => use best meta-learner's final accuracy & F1 from earlier
acc_comb = float(best_model_row['Test_Accuracy'])
f1_comb = float(best_model_row['Test_F1'])

print(f"Combined => Accuracy={acc_comb:.3f}, F1={f1_comb:.3f}")

# 5) Make a grouped bar plot: x-axis => [SNN, CAE, Combined]
#    We'll have 2 bars per group => Accuracy, F1
embedding_labels = ["SNN", "CAE", "Combined"]
acc_values = [acc_snn, acc_cae, acc_comb]
f1_values = [f1_snn, f1_cae, f1_comb]

x = np.arange(len(embedding_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))

# Accuracy bars (blue)
bar_acc = ax.bar(x - width / 2, acc_values, width=width,
                 label="Accuracy", color="blue")
# F1 bars (red)
bar_f1 = ax.bar(x + width / 2, f1_values, width=width,
                label="F1 Score", color="red")

# Y-limit from 0.75 to 1
ax.set_ylim([0.75, 1.0])

ax.set_xticks(x)
ax.set_xticklabels(embedding_labels, fontsize=10)
ax.set_ylabel("Metric Value", fontsize=11)
ax.set_xlabel("Embeddings Used", fontsize=11)
ax.set_title("Informativeness of Generated Embeddings", fontsize=12)

# Put legend outside on the right
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Annotate bars
for rect in bar_acc + bar_f1:
    height = rect.get_height()
    ax.annotate(f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # offset in points
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
out_barplot = os.path.join(evaluation_dir, "meta_embedding_comparison_barplot.png")
plt.savefig(out_barplot, dpi=300, bbox_inches='tight')
plt.close()

print(f"Bar plot saved => {out_barplot}")
print("Done comparing single-embedding vs combined embeddings.\n")

###############################################################################
# Centroids-Only 4-Axis Plots (Non-Met PRE/POST, Met PRE/POST) for PCA or tSNE,
# in 2D or 3D. 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os


def _transform_dimred(X, method='pca', is_3d=False, random_state=42):
    """
    Dimensionality reduction of X into 2D/3D using PCA or tSNE.
      X : (n_samples, n_features)
      method : 'pca' or 'tsne'
      is_3d  : bool -> produce 2 or 3 components
    Returns coords: (n_samples, 2 or 3).
    """
    n_components = 3 if is_3d else 2
    if method.lower() == 'pca':
        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(X)
    else:
        tsne = TSNE(n_components=n_components, random_state=random_state)
        coords = tsne.fit_transform(X)
    return coords


def _plot_centroids_only(ax, coords, labels, unique_labels, is_3d=False):
    """
    Plots only the centroids (one dot per label). Also computes distortion
    (mean distance from each label's points to its centroid) and returns
    the median distortion across all labels.
    No textual annotation on the axes themselves.
    """
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {ul: palette[i] for i, ul in enumerate(unique_labels)}

    # Store centroid and distortion
    label_centroids = {}
    distortions = []

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) == 0:
            continue
        subset = coords[idx]
        centroid = np.mean(subset, axis=0)
        label_centroids[lbl] = centroid

        # Distortion => mean distance to centroid
        dists = np.linalg.norm(subset - centroid, axis=1)
        distortions.append(np.mean(dists))

    # Plot centroids
    for lbl in unique_labels:
        ctd = label_centroids.get(lbl, None)
        if ctd is not None:
            if is_3d:
                ax.scatter(
                    ctd[0], ctd[1], ctd[2],
                    s=80, marker='o', alpha=0.9,
                    color=color_map[lbl], edgecolor='k',
                    label=lbl
                )
            else:
                ax.scatter(
                    ctd[0], ctd[1],
                    s=80, marker='o', alpha=0.9,
                    color=color_map[lbl], edgecolor='k',
                    label=lbl
                )

    if len(distortions) == 0:
        return 0.0
    return float(np.median(distortions))


def plot_4axes_centroids(X_pre_nonmet, X_pre_met,
                         X_post_nonmet, X_post_met,
                         labels_pre_nonmet, labels_pre_met,
                         labels_post_nonmet, labels_post_met,
                         unique_labels,
                         method='pca',
                         is_3d=False,
                         fig_title="",
                         out_png="centroids_4axes.png"):
    """
    2×2 centroid-only grids.  Legend removed to keep the figure clean.
    Each axis still shows its own median distortion.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib as mpl

    def _transform(X):
        n = 3 if is_3d else 2
        if method.lower() == 'pca':
            return PCA(n_components=n, random_state=42).fit_transform(X)
        return TSNE(n_components=n, random_state=42).fit_transform(X)

    def _median_dist(pts, lbls):
        vals = []
        for ul in unique_labels:
            idx = np.where(lbls == ul)[0]
            if idx.size:
                centre = pts[idx].mean(axis=0)
                vals.append(np.mean(np.linalg.norm(pts[idx] - centre, axis=1)))
        return np.median(vals) if vals else np.nan

    def _plot(ax, pts, lbls, title):
        palette = sns.color_palette("hls", len(unique_labels))
        color_map = {ul: palette[i] for i, ul in enumerate(unique_labels)}
        for ul in unique_labels:
            idx = np.where(lbls == ul)[0]
            if idx.size:
                if is_3d:
                    ax.scatter(*pts[idx].T, s=80, color=color_map[ul], edgecolor='k')
                else:
                    ax.scatter(pts[idx, 0], pts[idx, 1],
                               s=80, color=color_map[ul], edgecolor='k')
        ax.set_title(title, fontsize=mpl.rcParams['axes.titlesize'])
        md = _median_dist(pts, lbls)
        if is_3d:
            ax.text2D(0.97, 0.97, f"Dist={md:.2f}",
                      transform=ax.transAxes,
                      ha='right', va='top',
                      fontsize=mpl.rcParams['axes.labelsize'])
        else:
            ax.text(0.97, 0.97, f"Dist={md:.2f}",
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=mpl.rcParams['axes.labelsize'])

    fig = plt.figure(figsize=(14, 10))
    ax00 = fig.add_subplot(2, 2, 1, projection='3d' if is_3d else None)
    ax01 = fig.add_subplot(2, 2, 2, projection='3d' if is_3d else None)
    ax10 = fig.add_subplot(2, 2, 3, projection='3d' if is_3d else None)
    ax11 = fig.add_subplot(2, 2, 4, projection='3d' if is_3d else None)

    _plot(ax00, _transform(X_pre_nonmet), labels_pre_nonmet, "Non-Met PRE")
    _plot(ax01, _transform(X_post_nonmet), labels_post_nonmet, "Non-Met POST")
    _plot(ax10, _transform(X_pre_met), labels_pre_met, "Met PRE")
    _plot(ax11, _transform(X_post_met), labels_post_met, "Met POST")

    fig.suptitle(fig_title, fontsize=mpl.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# MONTE CARLO SIMULATION: "Killing" each gene by adding substantial Gaussian noise
# and observing how the top meta-learner's accuracy changes on the held-out test set.
# ALSO computing accuracy drops for each original LABEL (e.g., "Brain_GTEX", "PAAD_METASTATIC").
#
# We'll run this process num_monte_carlo_cycles times. After each run, we'll save
# a CSV (e.g. "monte_carlo_gene_importance_1st.csv", "2nd.csv", etc.) containing
# columns like:
#     Gene, AccuracyDrop, AccuracyDrop_Brain_GTEX, AccuracyDrop_PAAD_METASTATIC, ...
#
# Then, we will simply average each column per Gene across the runs to produce
# one final aggregated CSV (monte_carlo_gene_importance_aggregated.csv).
# -------------------------------------------------------------------------
from scipy.stats import rankdata

print("\n--- Starting Monte Carlo Gene 'Kill' Simulation (Best Meta-Learner Only) ---")

# 1) Identify the single best meta-learner from 'best_two_models' (the top row)
best_model_row = best_two_models.iloc[0]  # pick the first (best) row
best_iteration = int(best_model_row['Iteration'])
h1 = best_model_row['h1']
h2 = best_model_row['h2']
h3 = best_model_row['h3']
dr = best_model_row['dropout_rate']
l2r = best_model_row['l2_reg']
opt = best_model_row['optimizer']
bsize = int(best_model_row['batch_size'])
eps = int(best_model_row['epochs'])

print(f"Using meta-learner from iteration={best_iteration} with Test_Accuracy={best_model_row['Test_Accuracy']:.4f}")

# 2) Rebuild & re-train that top meta-learner on meta_feats_base_train (80%)
meta_model = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=meta_feats_base_train.shape[1],
    num_classes=len(met_class_labels)
)
meta_model.fit(meta_feats_base_train, y_base_train,
               validation_data=(meta_feats_meta_val, y_meta_val),
               epochs=eps, batch_size=bsize, verbose=0)

# 3) Compute baseline accuracy on the original test set (meta_feats_test)
y_pred_test_baseline = meta_model.predict(meta_feats_test).argmax(axis=1)
acc_baseline = accuracy_score(y_met_test, y_pred_test_baseline)
print(f"Baseline test accuracy (no noise) = {acc_baseline:.4f}")

# -------------------------------------------------------------------------
# 3a) Additionally, compute baseline accuracy *per label*, e.g., "Brain_GTEX",
#     "PAAD_METASTATIC", etc.
# -------------------------------------------------------------------------
test_data_filtered_df = test_data_scaled_for_emb.copy().reset_index(drop=True)

# We'll produce baseline predictions on this test set for each label
test_X_snn = snn_base_network.predict(test_data_filtered_df[fcols].values)
test_X_cae = cae_encoder.predict(test_data_filtered_df[fcols].values)
test_combined_emb = np.concatenate([test_X_snn, test_X_cae], axis=1)
test_meta_feats = get_meta_features(met_classifiers, test_combined_emb)
test_y_pred = meta_model.predict(test_meta_feats).argmax(axis=1)

unique_labels_for_acc = test_data_filtered_df['LABEL'].unique()
baseline_label_acc = {}
for lbl in unique_labels_for_acc:
    idx_ = (test_data_filtered_df['LABEL'] == lbl)
    if np.sum(idx_) == 0:
        continue
    baseline_label_acc[lbl] = accuracy_score(y_met_test[idx_], test_y_pred[idx_])

# -------------------------------------------------------------------------
# 4) We'll "kill" each gene by adding big Gaussian noise (noise_std=20)
#    and measure accuracy drops, repeating for num_monte_carlo_cycles times.
# -------------------------------------------------------------------------
num_monte_carlo_cycles = 10  # Adjust how many times you want to repeat
print(f"\nRepeating the Monte Carlo simulation {num_monte_carlo_cycles} times...")


def get_predictions_after_noise(test_df_noisy):
    """
    Given a test dataframe with noise in one gene, do:
      1) Re-embed with SNN & CAE
      2) get meta-features from base classifiers
      3) get predictions from meta_model
      4) return integer predicted labels
    """
    X_snn_noisy = snn_base_network.predict(test_df_noisy[fcols].values)
    X_cae_noisy = cae_encoder.predict(test_df_noisy[fcols].values)
    combined_emb_noisy = np.concatenate([X_snn_noisy, X_cae_noisy], axis=1)
    meta_feats_noisy = get_meta_features(met_classifiers, combined_emb_noisy)
    y_pred_noisy = meta_model.predict(meta_feats_noisy).argmax(axis=1)
    return y_pred_noisy


# We'll store the dataframes for each simulation in this list:
list_of_dfs = []

for cycle_idx in range(1, num_monte_carlo_cycles + 1):
    print(f"\n--- Monte Carlo Cycle {cycle_idx} / {num_monte_carlo_cycles} ---")

    # For each gene, add noise and measure accuracy drop
    noise_std = 20.0
    all_genes = list(feature_columns)
    gene_importance_results = []

    for gene_name in all_genes:
        # Copy of the original test set
        test_noisy_df = test_data_filtered_df.copy()

        # Add Gaussian noise to just this gene
        test_noisy_df[gene_name] += np.random.normal(loc=0.0, scale=noise_std, size=len(test_noisy_df))

        # Measure new overall accuracy
        y_pred_noisy = get_predictions_after_noise(test_noisy_df)
        new_acc = accuracy_score(y_met_test, y_pred_noisy)
        drop_in_acc = acc_baseline - new_acc

        # Measure per-label accuracy drops
        label_drop_dict = {}
        for lbl in unique_labels_for_acc:
            idx_lbl = (test_noisy_df['LABEL'] == lbl)
            if np.sum(idx_lbl) == 0:
                continue
            new_acc_lbl = accuracy_score(y_met_test[idx_lbl], y_pred_noisy[idx_lbl])
            drop_in_acc_lbl = baseline_label_acc[lbl] - new_acc_lbl
            label_drop_dict[lbl] = drop_in_acc_lbl

        # Build the row
        row_dict = {'Gene': gene_name, 'AccuracyDrop': drop_in_acc}
        for lbl in unique_labels_for_acc:
            if lbl in label_drop_dict:
                row_dict[f"AccuracyDrop_{lbl}"] = label_drop_dict[lbl]
            else:
                row_dict[f"AccuracyDrop_{lbl}"] = np.nan

        gene_importance_results.append(row_dict)

    # Create a DataFrame for this cycle
    df_cycle = pd.DataFrame(gene_importance_results)
    df_cycle.sort_values('AccuracyDrop', ascending=False, inplace=True)

    # rank-based p-value for the cycle
    all_drops = df_cycle['AccuracyDrop'].values
    ranks_desc = rankdata(-all_drops, method='average')  # descending rank
    p_vals = ranks_desc / (len(all_drops) + 1.0)
    df_cycle['p_value'] = p_vals

    # Save CSV with a suffix (1st, 2nd, etc.)
    suffix_str = f"{cycle_idx}th"
    cycle_csv_name = f"monte_carlo_gene_importance_{suffix_str}.csv"
    cycle_csv_path = os.path.join(evaluation_dir, cycle_csv_name)
    df_cycle.to_csv(cycle_csv_path, index=False)
    print(f"Cycle {cycle_idx} => saved CSV to '{cycle_csv_name}'")

    # Store this DataFrame for averaging
    # IMPORTANT: We'll sort by 'Gene' so each cycle's rows are in the same order
    #            This ensures a straightforward average across all cycles.
    df_cycle_sorted = df_cycle.sort_values('Gene').reset_index(drop=True)
    list_of_dfs.append(df_cycle_sorted)

# -------------------------------------------------------------------------
# 5) Average across the cycles: for each gene, each column => mean
#    We'll assume all cycles have the same set of Genes in the same order
#    once sorted by 'Gene'.
# -------------------------------------------------------------------------
print("\nAveraging results across all Monte Carlo cycles...")

df_final = list_of_dfs[0].copy()
num_cycles = len(list_of_dfs)

for col in df_final.columns:
    if col == 'Gene':
        continue
    # We'll accumulate the values from each cycle
    # Start from cycle #2 in our list_of_dfs
    for c_idx in range(1, num_cycles):
        df_final[col] += list_of_dfs[c_idx][col]
    # Now divide
    df_final[col] /= float(num_cycles)

# Recompute p_value as a rank-based measure on the newly averaged 'AccuracyDrop'
all_avg_drops = df_final['AccuracyDrop'].values
ranks_desc = rankdata(-all_avg_drops, method='average')  # descending rank
p_vals = ranks_desc / (len(all_avg_drops) + 1.0)
df_final['p_value'] = p_vals

# Sort by 'AccuracyDrop' descending
df_final.sort_values('AccuracyDrop', ascending=False, inplace=True)

# Save final aggregated CSV
final_csv_path = os.path.join(evaluation_dir, "monte_carlo_gene_importance_aggregated.csv")
df_final.to_csv(final_csv_path, index=False)
print(f"Final aggregated CSV saved => {final_csv_path}")

# 6) Plot top-20 by the new average AccuracyDrop
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

print(f"Aggregated top-20 bar plot saved => {plot_path}")
print("Monte Carlo gene-kill simulation complete (best meta-learner). Ready for unit tests.")

print("All evaluations completed. Results and plots are saved in the 'TestEvaluation' directory.")
print("Running unit tests on splits:")
unittest.main(argv=[''], exit=False)
print("Script execution completed.")

