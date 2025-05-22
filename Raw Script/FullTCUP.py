import os
import unittest
import pandas as pd
import numpy as np
import random
import pickle
import glob
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
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

_BASE     = 10        # master knob (was 20)
_TICKS    = _BASE - 7
_LEGEND   = _BASE - 5

plt.rcParams.update({
    'font.size'        : _BASE,
    'axes.titlesize'   : _BASE + 2,
    'axes.labelsize'   : _BASE,
    'xtick.labelsize'  : _TICKS,
    'ytick.labelsize'  : _TICKS,
    'legend.fontsize'  : _LEGEND,
    'figure.titlesize' : _BASE + 4,
    'figure.dpi'       : 300,
})

# Make seaborn honour the same scale
sns.set_context("notebook", font_scale=_BASE / 10)

np.random.seed(40)
random.seed(40)
tf.random.set_seed(40)

project_dir = '.'
patient_data_file = os.path.join(project_dir, "BalancedTCGA_CancerTranscriptomics.csv")
gtex_reads_dir = os.path.join(project_dir, "GTEx Reads")
ccle_expr_file = os.path.join(project_dir, "CCLE_RNAseq_readsParsed.csv")
ccle_clinical_file = os.path.join(project_dir, "ccle_broad_2019_clinical_data.csv")
metastatic_data_file = os.path.join(project_dir, "metastatic_TOO_dataset_tpm.csv")

output_base_dir = os.path.join(project_dir, "MetastaticTOO_IncorpTest")
os.makedirs(output_base_dir, exist_ok=True)
gtex_processed_dir = os.path.join(project_dir, "GTExProcessed_noZ")
os.makedirs(gtex_processed_dir, exist_ok=True)

new_output_dir = os.path.join(output_base_dir, "Full_wFullGenesModifiedMaxPairsv2")
os.makedirs(new_output_dir, exist_ok=True)
merged_data_preprocessed_file = os.path.join(new_output_dir, "merged_data_preprocessed.pkl")
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
    Loads or processes GTEx data and returns gtex_robust, gtex_all
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

# ---------- Pair Creation Logic ------------
def create_pairs_by_logic(data,
                          max_pairs=40000,
                          max_usage_per_labelpos=max_usage_per_sample_posneg,
                          max_usage_per_labelneg=max_usage_per_sample_posneg
                          ,
                          overrepresent_metastatic=False):
    """
    Creates positive and negative pairs from 'data'.
    Up to max_usage_per_labelpos usage for positive pairs,
    up to max_usage_per_labelneg usage for negative pairs.
    Overrepresent metastatic if needed.
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
        random.shuffle(indices_lbl)
        for i in range(len(indices_lbl)):
            for j in range(i+1, len(indices_lbl)):
                if (usage_counts_pos[indices_lbl[i]] < max_usage_per_labelpos and
                    usage_counts_pos[indices_lbl[j]] < max_usage_per_labelpos):
                    positive_pairs.append((indices_lbl[i], indices_lbl[j]))
                    usage_counts_pos[indices_lbl[i]] += 1
                    usage_counts_pos[indices_lbl[j]] += 1
                    if len(positive_pairs) >= max_pairs//2:
                        break
            if len(positive_pairs) >= max_pairs//2:
                break
        if len(positive_pairs) >= max_pairs//2:
            break

    # 2) Negative pairs
    label_combos = list(combinations(labels, 2))
    for (l1, l2) in label_combos:
        idx1 = list(label_indices[l1])
        idx2 = list(label_indices[l2])
        random.shuffle(idx1)
        random.shuffle(idx2)
        for a in idx1:
            for b in idx2:
                if (usage_counts_neg[a] < max_usage_per_labelneg and
                    usage_counts_neg[b] < max_usage_per_labelneg):
                    negative_pairs.append((a, b))
                    usage_counts_neg[a] += 1
                    usage_counts_neg[b] += 1
                    if len(negative_pairs) >= max_pairs//2:
                        break
            if len(negative_pairs) >= max_pairs//2:
                break
        if len(negative_pairs) >= max_pairs//2:
            break

    # 3) Overrepresent metastatic if requested
    if overrepresent_metastatic:
        meta_counts_pos = defaultdict(int)
        meta_pos = []
        for i in range(len(metastatic_indices)):
            for j in range(i+1, len(metastatic_indices)):
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
    all_labels = [1]*len(positive_pairs) + [0]*len(negative_pairs)
    all_pairs, all_labels = shuffle(all_pairs, all_labels, random_state=42)
    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs[:max_pairs]
        all_labels = all_labels[:max_pairs]

    return np.array(all_pairs), np.array(all_labels)

# ---------- PairGenerator ------------
class PairGenerator(Sequence):
    """
    Generates pairs (X1, X2) for training/eval.
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
        return int(np.ceil(len(self.pairs)/self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_pairs = self.pairs[batch_indices]
        batch_labels = self.labels[batch_indices]
        X1_batch = self.data.iloc[batch_pairs[:,0]][self.feature_columns].values.astype(np.float32)
        X2_batch = self.data.iloc[batch_pairs[:,1]][self.feature_columns].values.astype(np.float32)
        if self.augment:
            X1_batch += np.random.normal(0, 0.01, X1_batch.shape)
            X2_batch += np.random.normal(0, 0.01, X2_batch.shape)
        return [X1_batch, X2_batch], batch_labels

# ---------- Model Builders ------------
def create_modified_contrastive_autoencoder(input_shape):
    """
    Contrastive AE: encoder + decoder, final latent is L2-normalized
    """
    initializer = tf.keras.initializers.HeNormal()

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
    Creates a base MLP network for Siamese (SNN)
    """
    initializer = tf.keras.initializers.HeNormal()
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
    Identify top-k% "hard" SNN examples by single-sample BCE + measure.
    'uncertainty' => hardness = BCE + (1 - |pred-0.5|)
    'overconfident' => hardness = BCE + |pred-0.5|
    """
    import math

    def single_sample_bce(y_true, y_pred, eps=1e-7):
        y_clamped = max(min(float(y_pred), 1-eps), eps)
        return - (float(y_true)*math.log(y_clamped) +
                  (1.0 - float(y_true))*math.log(1-y_clamped))

    hardness_values = []
    indices = []

    for batch_idx in range(len(data_generator)):
        (X1_batch, X2_batch), y_batch = data_generator[batch_idx]
        y_pred = model.predict([X1_batch, X2_batch], verbose=0).ravel()
        start_idx = batch_idx * data_generator.batch_size
        end_idx   = start_idx + len(y_batch)
        batch_ind = data_generator.indices[start_idx:end_idx]

        for i in range(len(y_batch)):
            lbl = float(y_batch[i])
            pp  = float(y_pred[i])
            bce = single_sample_bce(lbl, pp)
            if mode.lower()=='uncertainty':
                measure = 1.0 - abs(pp-0.5)
            else:
                measure = abs(pp-0.5)
            hardness = bce + measure
            hardness_values.append(hardness)

        indices.extend(batch_ind)

    hardness_values = np.array(hardness_values)
    indices = np.array(indices)
    sorted_desc = np.argsort(hardness_values)[::-1]
    num_samples = int(top_k_percent*len(hardness_values))
    if min_num_samples is not None:
        num_samples = max(num_samples, min_num_samples)
    num_samples = min(num_samples, len(hardness_values))

    hard_inds = sorted_desc[:num_samples]
    if mode.lower()=='overconfident':
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
            return dist**2
        else:
            return max(margin - dist, 0.0)**2

    def single_sample_mse(x_true, x_dec):
        """Mean squared error for one sample reconstruction."""
        return np.mean((x_true - x_dec)**2)

    hardness_values = []
    indices = []

    for batch_idx in range(len(data_generator)):
        # Our generator yields: X=(X1_batch,X2_batch), y=(y_contrast,X1_orig,X2_orig)
        (X1_batch, X2_batch), y_batch = data_generator[batch_idx]
        y_contrast_batch = y_batch[0]  # shape (batch_size,1) typically
        X1_orig_batch    = y_batch[1]  # shape (batch_size, input_dim)
        X2_orig_batch    = y_batch[2]  # shape (batch_size, input_dim)

        # Model outputs 3 arrays: distz, decA, decB
        distz_batch, decA_batch, decB_batch = model.predict([X1_batch, X2_batch], verbose=0)

        # Identify indices in the entire dataset
        start_idx = batch_idx * data_generator.batch_size
        end_idx   = start_idx + len(X1_batch)
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
    plt.figure(figsize=(12,10))
    colors = sns.color_palette("hls", len(classes))
    for i in range(len(classes)):
        if np.sum(y_true_binarized[:, i])==0:
            continue
        fpr, tpr, _ = roc_curve(y_true_binarized[:,i], y_pred_proba[:,i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label='{0} (AUC={1:0.2f})'.format(classes[i], roc_auc_val))
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([-0.05,1.05])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.3),
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
    val_loss   = history.history.get('val_loss')
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss,  'r-', label='Val Loss')

    # vertical markers
    plt.axvline(x=uncertainty_start + 0.5,    color='red',   ls=':',  label='Uncertainty HEM')
    plt.axvline(x=overconfidence_start + 0.5, color='gold',  ls='--', lw=2, label='Overconfidence HEM')
    plt.axvline(x=end_hem + 0.5,              color='black', ls=':',  label='All HEM End')

    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # <<<  larger legend
    big_font = mpl.rcParams['axes.titlesize']
    plt.legend(loc='upper right', fontsize=big_font)

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def evaluate_snn(model, generator, output_dir):
    """
    Confusion-matrix + ROC side-by-side, with legend outside so it never overlaps.
    """
    all_labels, all_preds = [], []
    for idx in range(len(generator)):
        (X1, X2), y = generator[idx]
        all_preds.extend(model.predict([X1, X2], verbose=0).ravel())
        all_labels.extend(y)

    all_labels = np.asarray(all_labels, int)
    all_preds  = np.asarray(all_preds)

    bin_preds = (all_preds >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, bin_preds, labels=[0, 1])
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))   # a bit smaller

    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                xticklabels=['Pred:0', 'Pred:1'],
                yticklabels=['True:0', 'True:1'],
                ax=ax[0])
    ax[0].set_title("SNN Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    ax[1].plot([0, 1], [0, 1], ls='--', color='grey')
    ax[1].set_title("SNN ROC")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'snn_confusion_and_roc.png'), dpi=300)
    plt.close(fig)

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
    mse_vals = np.mean((X_all - X_dec_all)**2, axis=1)
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
        X_dec  = X_dec_all[idx_]

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


def plot_dim_reductions_2d(X_pre, X_post, labels, set_name, output_dir, unique_labels, hide_legend=False):
    """
    Creates a single figure for 2D TSNE, 2D PCA, 2D UMAP (pre vs. post).
    We'll do side-by-side subplots for each method:
      top row = pre, bottom row = post
      columns = TSNE, PCA, UMAP
    If hide_legend=True, we skip the final fig.legend(...) call.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # TSNE 2D
    tsne_pre  = TSNE(n_components=2, random_state=42).fit_transform(X_pre)
    tsne_post = TSNE(n_components=2, random_state=42).fit_transform(X_post)
    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    pca_pre  = pca.fit_transform(X_pre)
    pca_post = pca.fit_transform(X_post)
    # UMAP 2D
    umap_model = UMAP(n_components=2, random_state=42)
    umap_pre  = umap_model.fit_transform(X_pre)
    umap_post = umap_model.fit_transform(X_post)

    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    def scatter_2d(ax, coords, lbls, title):
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=[color_map[lbl]], label=str(lbl), s=20, alpha=0.6)
        ax.set_title(title)

    # row0 col0 => TSNE pre
    scatter_2d(axes[0], tsne_pre, labels, "t-SNE (Pre)")
    # row0 col1 => PCA pre
    scatter_2d(axes[1], pca_pre, labels, "PCA (Pre)")
    # row0 col2 => UMAP pre
    scatter_2d(axes[2], umap_pre, labels, "UMAP (Pre)")
    # row1 col0 => TSNE post
    scatter_2d(axes[3], tsne_post, labels, "t-SNE (Post)")
    # row1 col1 => PCA post
    scatter_2d(axes[4], pca_post, labels, "PCA (Post)")
    # row1 col2 => UMAP post
    scatter_2d(axes[5], umap_post, labels, "UMAP (Post)")

    handles, lbls = axes[0].get_legend_handles_labels()
    if not hide_legend:
        fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                   ncol=4, fontsize='small')

    fig.suptitle(f"{set_name} 2D Dimensionality Reduction", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"{set_name}_2D_dim_reduction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_dim_reductions_3d(X_pre, X_post, labels, set_name, output_dir, unique_labels, hide_legend=False):
    """
    Creates a single figure for 3D TSNE, 3D PCA, 3D UMAP (pre vs. post).
    We'll do side-by-side subplots for each method:
      top row = Pre, bottom row = Post
      columns = TSNE, PCA, UMAP
    If hide_legend=True, we skip the final fig.legend(...) call.
    """
    fig = plt.figure(figsize=(15, 9))

    # TSNE 3D
    tsne_pre  = TSNE(n_components=3, random_state=42).fit_transform(X_pre)
    tsne_post = TSNE(n_components=3, random_state=42).fit_transform(X_post)
    # PCA 3D
    pca = PCA(n_components=3, random_state=42)
    pca_pre  = pca.fit_transform(X_pre)
    pca_post = pca.fit_transform(X_post)
    # UMAP 3D
    umap_model = UMAP(n_components=3, random_state=42)
    umap_pre  = umap_model.fit_transform(X_pre)
    umap_post = umap_model.fit_transform(X_post)

    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # row0: TSNE / PCA / UMAP
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    # row1: TSNE / PCA / UMAP
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')

    def scatter_3d(ax, coords, lbls, ttl):
        for lbl in unique_labels:
            idx = np.where(lbls == lbl)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                       c=[color_map[lbl]], label=str(lbl), s=20, alpha=0.6)
        ax.set_title(ttl)

    # row0 col0 => TSNE pre
    scatter_3d(ax1, tsne_pre,  labels, "t-SNE (Pre)")
    scatter_3d(ax2, pca_pre,   labels, "PCA (Pre)")
    scatter_3d(ax3, umap_pre,  labels, "UMAP (Pre)")

    # row1 col0 => TSNE post
    scatter_3d(ax4, tsne_post, labels, "t-SNE (Post)")
    scatter_3d(ax5, pca_post,  labels, "PCA (Post)")
    scatter_3d(ax6, umap_post, labels, "UMAP (Post)")

    handles, lbls = ax1.get_legend_handles_labels()
    if not hide_legend:
        fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                   ncol=4, fontsize='small')

    fig.suptitle(f"{set_name} 3D Dimensionality Reduction", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(output_dir, f"{set_name}_3D_dim_reduction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------- MAIN SCRIPT START ----------------
print("Loading and preprocessing data...")

if os.path.exists(merged_data_preprocessed_file):
    with open(merged_data_preprocessed_file, 'rb') as f:
        pre_data = pickle.load(f)
    merged_data = pre_data['merged_data']
    feature_columns = pre_data['common_genes']
    train_data = pre_data['train_data']
    val_data = pre_data['val_data']
    test_data = pre_data['test_data']
    label_encoder = pre_data['label_encoder']
    print("Preprocessed data loaded.")
else:
    print("Preprocessed data not found. Starting data loading and preprocessing...")
    patient_data = pd.read_csv(patient_data_file)
    metastatic_data = pd.read_csv(metastatic_data_file)
    gtex_robust, _ = process_gtex_files(gtex_reads_dir, gtex_processed_dir)

    # Convert columns to uppercase
    patient_data.columns = patient_data.columns.str.upper()
    gtex_robust.columns = gtex_robust.columns.str.upper()
    metastatic_data.columns = metastatic_data.columns.str.upper()

    # Identify gene columns
    non_gene_columns_patient = ['CANCER TYPE', 'CLASS']
    gene_columns_patient = [col for col in patient_data.columns if col not in non_gene_columns_patient]

    non_gene_columns_gtex_robust = ['TISSUE_ROBUST']
    gene_columns_gtex_robust = [col for col in gtex_robust.columns if col not in non_gene_columns_gtex_robust]

    non_gene_columns_metastatic = ['CANCERTYPE']
    gene_columns_metastatic = [col for col in metastatic_data.columns if col not in non_gene_columns_metastatic]

    # We skip CCLE entirely, so let's gather common genes from TCGA + GTEX + MET
    common_genes = list(
        set(gene_columns_patient)
        & set(gene_columns_gtex_robust)
        & set(gene_columns_metastatic)
    )

    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)

    # -------------------
    # 1) Impute TCGA (patient)
    # -------------------
    patient_expr = patient_data[common_genes]
    patient_expr_imputed = imputer.fit_transform(patient_expr)
    patient_expr_imputed = pd.DataFrame(patient_expr_imputed, columns=common_genes)

    # -------------------
    # 2) Impute GTEX
    # -------------------
    gtex_expr = gtex_robust[common_genes]
    gtex_expr_imputed = imputer.transform(gtex_expr)
    gtex_expr_imputed = pd.DataFrame(gtex_expr_imputed, columns=common_genes)

    # -------------------
    # 3) Impute Metastatic
    # -------------------
    metastatic_expr = metastatic_data[common_genes]
    metastatic_expr_imputed = imputer.transform(metastatic_expr)
    metastatic_expr_imputed = pd.DataFrame(metastatic_expr_imputed, columns=common_genes)

    # Assign SOURCE, LABEL, and TISSUE columns
    patient_labels = patient_data['CANCER TYPE'].reset_index(drop=True)
    gtex_labels = gtex_robust['TISSUE_ROBUST'].reset_index(drop=True)
    metastatic_labels = metastatic_data['CANCERTYPE'].reset_index(drop=True)

    patient_expr_imputed.index = patient_labels.index
    gtex_expr_imputed.index = gtex_labels.index
    metastatic_expr_imputed.index = metastatic_labels.index

    patient_expr_imputed['SOURCE'] = 'TCGA'
    gtex_expr_imputed['SOURCE'] = 'GTEX'
    metastatic_expr_imputed['SOURCE'] = 'METASTATIC'

    patient_expr_imputed['LABEL'] = patient_labels.values + '_TCGA'
    gtex_expr_imputed['LABEL'] = gtex_labels.values + '_GTEX'
    metastatic_expr_imputed['LABEL'] = metastatic_labels.values + '_METASTATIC'
    metastatic_expr_imputed['TISSUE'] = metastatic_labels.values

    # -------------------
    # 4) Apply "≥30" filter only to TCGA+GTEX
    # -------------------
    patient_gtex_merged = pd.concat([patient_expr_imputed, gtex_expr_imputed], ignore_index=True)
    pg_label_counts = patient_gtex_merged['LABEL'].value_counts()
    pg_sufficient_labels = pg_label_counts[pg_label_counts >= 30].index.tolist()

    # Keep only those "≥30" labels in patient+gtex
    patient_gtex_merged = patient_gtex_merged[patient_gtex_merged['LABEL'].isin(pg_sufficient_labels)].reset_index(drop=True)

    # Re-split back into patient/gtex
    patient_expr_imputed = patient_gtex_merged[patient_gtex_merged['SOURCE'] == 'TCGA'].copy().reset_index(drop=True)
    gtex_expr_imputed    = patient_gtex_merged[patient_gtex_merged['SOURCE'] == 'GTEX'].copy().reset_index(drop=True)

    # -------------------
    # 5) Filter out metastatic labels that have <5 samples
    # -------------------
    meta_label_counts = metastatic_expr_imputed['LABEL'].value_counts()
    # keep only those MET labels with >=5
    keep_met_labels = meta_label_counts[meta_label_counts >= 5].index.tolist()
    metastatic_expr_imputed = metastatic_expr_imputed[metastatic_expr_imputed['LABEL'].isin(keep_met_labels)].reset_index(drop=True)

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
            random_state=42,
            stratify=metastatic_expr_imputed['LABEL']
        )
    else:
        train_meta = pd.DataFrame(columns=metastatic_expr_imputed.columns)
        val_meta   = pd.DataFrame(columns=metastatic_expr_imputed.columns)

    # -------------------
    # 8) Merge patient+gtex => main splits (60:20:20)
    # -------------------
    pg_merged_filtered = pd.concat([patient_expr_imputed, gtex_expr_imputed], ignore_index=True)
    X_full = pg_merged_filtered
    y_full = pg_merged_filtered['LABEL']

    train_main, temp_main = train_test_split(
        X_full, test_size=0.4, random_state=42, stratify=y_full
    )
    y_temp = temp_main['LABEL']
    val_main, test_main = train_test_split(
        temp_main, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Final train/val/test
    train_data = pd.concat([train_main, train_meta], ignore_index=True)
    val_data   = pd.concat([val_main,   val_meta],   ignore_index=True)
    test_data  = pd.concat([test_main,  test_samples], ignore_index=True)

    # Ensure unique SAMPLE_ID
    if 'SAMPLE_ID' not in train_data.columns:
        train_data['SAMPLE_ID'] = [f"TRAIN_{i}" for i in range(len(train_data))]
    if 'SAMPLE_ID' not in val_data.columns:
        val_data['SAMPLE_ID'] = [f"VAL_{i}" for i in range(len(val_data))]
    if 'SAMPLE_ID' not in test_data.columns:
        test_data['SAMPLE_ID'] = [f"TEST_{i}" for i in range(len(test_data))]

    # Remove duplicates across splits by SAMPLE_ID
    train_ids = set(train_data['SAMPLE_ID'])
    val_data  = val_data[~val_data['SAMPLE_ID'].isin(train_ids)]
    val_ids   = set(val_data['SAMPLE_ID'])
    test_data = test_data[~test_data['SAMPLE_ID'].isin(train_ids)]
    test_data = test_data[~test_data['SAMPLE_ID'].isin(val_ids)]
    train_data = train_data.reset_index(drop=True)
    val_data   = val_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    # 9) Label-encode only what's left
    train_labels_set = set(train_data['LABEL'].unique())
    val_data  = val_data[val_data['LABEL'].isin(train_labels_set)].reset_index(drop=True)
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
    val_data['LABEL_NUMERIC']   = label_encoder.transform(val_data['LABEL'])
    test_data['LABEL_NUMERIC']  = label_encoder.transform(test_data['LABEL'])

    # 10) log2 transform on final sets
    for g in common_genes:
        train_data[g] = np.log2(train_data[g] + 1)
        val_data[g]   = np.log2(val_data[g] + 1)
        test_data[g]  = np.log2(test_data[g] + 1)

    # # 11) Standard scaling (optional: do it here or later)
    # scaler = StandardScaler()
    # scaler.fit(train_data[common_genes])
    # train_data[common_genes] = scaler.transform(train_data[common_genes])
    # val_data[common_genes]   = scaler.transform(val_data[common_genes])
    # test_data[common_genes]  = scaler.transform(test_data[common_genes])

    # 12) SMOTE on metastatic in train_data (only for labels with >=3 samples)
    from imblearn.over_sampling import SMOTE

    # Identify metastatic labels in train_data
    metastatic_labels_in_train = [
        lbl for lbl in train_data['LABEL'].unique()
        if lbl.endswith('_METASTATIC')
    ]

    label_counts_pre = train_data['LABEL'].value_counts().to_dict()
    smote_dict = {}
    for lbl in metastatic_labels_in_train:
        if label_counts_pre.get(lbl, 0) >= 3:
            # Example: double the count
            smote_dict[lbl] = label_counts_pre[lbl] * 2

    if len(smote_dict) > 0:
        sm = SMOTE(
            sampling_strategy=smote_dict,
            random_state=42,
            k_neighbors=2
        )
        X_sm = train_data[common_genes]
        y_sm = train_data['LABEL']

        mask_for_smote = train_data['LABEL'].isin(smote_dict.keys())
        X_smote_part   = X_sm[mask_for_smote]
        y_smote_part   = y_sm[mask_for_smote]

        X_not = X_sm[~mask_for_smote]
        y_not = y_sm[~mask_for_smote]

        X_res, y_res = sm.fit_resample(X_smote_part, y_smote_part)
        smote_df = pd.DataFrame(X_res, columns=common_genes)
        smote_df['LABEL'] = y_res.values

        train_data = pd.concat([
            pd.DataFrame(X_not, columns=common_genes).assign(LABEL=y_not.values),
            smote_df
        ], ignore_index=True)
    else:
        print("No metastatic labels had >=3 samples; skipping SMOTE on metastatic.")

    # Recompute label_counts after SMOTE
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
    # with columns: Label,TrainCount,ValCount,TestCount,SMOTE,SMOTE_Generated
    train_counts = train_data['LABEL'].value_counts().to_dict()
    val_counts   = val_data['LABEL'].value_counts().to_dict()
    test_counts  = test_data['LABEL'].value_counts().to_dict()

    # How many new samples were created by SMOTE for each label:
    # new_count = after_smote - before_smote
    smote_generated_dict = {}
    for lbl in train_counts.keys():
        before_ = label_counts_pre.get(lbl, 0)
        after_  = train_counts[lbl]
        diff_   = max(0, after_ - before_)
        smote_generated_dict[lbl] = diff_

    rows_list = []
    all_labels_final = set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys())
    for lbl in sorted(all_labels_final):
        row_ = {
            'Label': lbl,
            'TrainCount': train_counts.get(lbl, 0),
            'ValCount':   val_counts.get(lbl, 0),
            'TestCount':  test_counts.get(lbl, 0)
        }
        if (lbl in smote_dict):
            row_['SMOTE'] = True
            row_['SMOTE_Generated'] = smote_generated_dict.get(lbl, 0)
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

    # Finally, save pre_data
    with open(merged_data_preprocessed_file, 'wb') as f:
        pickle.dump({
            'merged_data': merged_data,
            'common_genes': common_genes,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'label_encoder': label_encoder
        }, f)
    print("Preprocessed data saved. Metastatic labels <5 removed, 4-class test extracted, leftover 75:25 splitted, SMOTE applied, CSV stored.")

    pre_data = {
        'merged_data': merged_data,
        'common_genes': common_genes,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'label_encoder': label_encoder
    }

print("Performing feature selection with Lasso (no CV) and limiting max genes...")

max_genes = 600 # <-- ADJUST THIS TO YOUR DESIRED MAX NUMBER OF GENES

if os.path.exists(feature_sets_file):
    with open(feature_sets_file, 'rb') as f:
        feature_sets = pickle.load(f)
else:
    train_data_local = pre_data['train_data']
    genes_for_selection = pre_data['common_genes']
    X_fs = train_data_local[genes_for_selection].values
    y_fs = train_data_local['LABEL_NUMERIC'].values

    # Lasso without cross-validation
    lasso_model = Lasso(alpha=0.001, random_state=42, max_iter=10000)

    # Use SelectFromModel to ensure no more than 'max_genes' are selected
    sfm = SelectFromModel(estimator=lasso_model,
                          max_features=max_genes,
                          threshold=-np.inf)  # -np.inf => select up to max_features

    sfm.fit(X_fs, y_fs)
    selected_mask = sfm.get_support()
    selected_genes = [g for g, use_g in zip(genes_for_selection, selected_mask) if use_g]

    # If no genes got selected, fall back to all genes (rare edge case)
    if len(selected_genes) == 0:
        selected_genes = genes_for_selection

    feature_sets = {'FullLasso': selected_genes}
    with open(feature_sets_file, 'wb') as f:
        pickle.dump(feature_sets, f)

feature_columns = feature_sets['FullLasso']
print("Number of features selected:", len(feature_columns))

print("Checking or creating scaled data with SMOTE if needed...")
if os.path.exists(merged_data_scaled_file):
    with open(merged_data_scaled_file, 'rb') as f:
        data_scaled = pickle.load(f)
    train_data_scaled = data_scaled['train_data_scaled']
    val_data_scaled   = data_scaled['val_data_scaled']
    test_data_scaled  = data_scaled['test_data_scaled']
    scaler = data_scaled['scaler']
else:
    train_data_local = pre_data['train_data']
    val_data_local   = pre_data['val_data']
    test_data_local  = pre_data['test_data']

    scaler = StandardScaler()
    scaler.fit(train_data_local[feature_columns])
    with open(scaler_file,'wb') as f:
        pickle.dump(scaler,f)

    train_data_scaled = train_data_local.copy()
    val_data_scaled   = val_data_local.copy()
    test_data_scaled  = test_data_local.copy()

    # ------------------------------
    #  Fix: Ensure SAMPLE_ID is in scaled data
    # ------------------------------
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

    # Now do standard scaling on the same columns
    train_data_scaled[feature_columns] = scaler.transform(train_data_local[feature_columns])
    val_data_scaled[feature_columns]   = scaler.transform(val_data_local[feature_columns])
    test_data_scaled[feature_columns]  = scaler.transform(test_data_local[feature_columns])

    # Attempt SMOTE on metastatic classes in train
    train_metas = train_data_scaled[train_data_scaled['LABEL'].str.contains('_METASTATIC')]
    class_counts = train_metas['LABEL'].value_counts()
    classes_for_smote = class_counts[class_counts>=5].index.tolist()
    if len(classes_for_smote)>0:
        sm = SMOTE(sampling_strategy={cls:int(class_counts[cls]*2)
                      for cls in classes_for_smote},
                   random_state=42, k_neighbors=2)
        X_sm = train_data_scaled[feature_columns]
        y_sm = train_data_scaled['LABEL']
        mask_for_smote = train_data_scaled['LABEL'].isin(classes_for_smote)
        X_smote_part = X_sm[mask_for_smote]
        y_smote_part = y_sm[mask_for_smote]
        X_not = X_sm[~mask_for_smote]
        y_not = y_sm[~mask_for_smote]
        X_res, y_res = sm.fit_resample(X_smote_part, y_smote_part)
        train_data_scaled = pd.concat([
            pd.DataFrame(X_not, columns=feature_columns).assign(
                LABEL=y_not.values,
                SOURCE=train_data_scaled[~mask_for_smote]['SOURCE'].values,
                LABEL_NUMERIC=train_data_scaled[~mask_for_smote]['LABEL_NUMERIC'].values,
                SAMPLE_ID=train_data_scaled[~mask_for_smote]['SAMPLE_ID'].values
            ),
            pd.DataFrame(X_res, columns=feature_columns).assign(
                LABEL=y_res.values
            )
        ], ignore_index=True)
        # fix columns for new SMOTE rows
        train_data_scaled['SOURCE'] = train_data_scaled['LABEL'].apply(
            lambda x: 'METASTATIC' if '_METASTATIC' in x else (
                'TCGA' if '_TCGA' in x else (
                    'GTEX' if '_GTEX' in x else 'UNKNOWN'
                )
            )
        )
        train_data_scaled['LABEL_NUMERIC'] = pre_data['label_encoder'].transform(train_data_scaled['LABEL'])

        # For newly created rows, if 'SAMPLE_ID' is empty, fill them
        for idx_, row_ in train_data_scaled.iterrows():
            if not row_.get('SAMPLE_ID', ''):
                train_data_scaled.at[idx_,'SAMPLE_ID'] = f"SMOTE_{idx_}"

        print("SMOTE was performed on metastatic classes in the train set.")

    val_data_scaled['LABEL_NUMERIC']  = pre_data['label_encoder'].transform(val_data_scaled['LABEL'])
    test_data_scaled['LABEL_NUMERIC'] = pre_data['label_encoder'].transform(test_data_scaled['LABEL'])

    train_data_scaled['SET'] = 'Train'
    val_data_scaled['SET']   = 'Validation'
    test_data_scaled['SET']  = 'Test'

    with open(merged_data_scaled_file, 'wb') as f:
        pickle.dump({
            'train_data_scaled': train_data_scaled,
            'val_data_scaled': val_data_scaled,
            'test_data_scaled': test_data_scaled,
            'scaler': scaler
        }, f)

print("Storing a CSV of how many samples per label in each split for documentation...")
split_label_counts = {}
for nm, dt in zip(['Train','Validation','Test'],
                  [train_data_scaled, val_data_scaled, test_data_scaled]):
    cts = dt['LABEL'].value_counts().to_dict()
    split_label_counts[nm] = cts

df_split_counts = pd.DataFrame(columns=['Label','TrainCount','ValCount','TestCount'])
all_lbls = (set(split_label_counts['Train'].keys())|
            set(split_label_counts['Validation'].keys())|
            set(split_label_counts['Test'].keys()))
for lbl in sorted(list(all_lbls)):
    row_ = {
        'Label': lbl,
        'TrainCount': split_label_counts['Train'].get(lbl,0),
        'ValCount':   split_label_counts['Validation'].get(lbl,0),
        'TestCount':  split_label_counts['Test'].get(lbl,0)
    }
    df_split_counts = df_split_counts.append(row_, ignore_index=True)
df_split_counts.to_csv(os.path.join(evaluation_dir,'split_label_counts.csv'), index=False)

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
    Evaluates the SNN on a held-out test set (pair classification).
    Produces a single figure with:
      - Left axis (ax[0]): Confusion Matrix for binary classification (0 vs. 1)
      - Right axis (ax[1]): ROC curve
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
                xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0','True:1'],
                ax=ax[0])
    ax[0].set_title("SNN Confusion Matrix (Test)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # (B) ROC on ax[1]
    ax[1].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc_val:.3f}")
    ax[1].plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
    ax[1].set_title("SNN ROC (Test)")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'snn_confusion_and_roc.png'), dpi=300)
    plt.close(fig)


# ------------------ SNN Training (40 epochs total) ------------------
if not os.path.exists(snn_model_file):
    print("Training SNN with overrepresented metastatic pairs...")

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
    snn_model = tf.keras.models.load_model(snn_model_file)


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
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_pairs = self.pairs[batch_indices]
        batch_labels = self.labels[batch_indices]

        X1_batch = self.data.iloc[batch_pairs[:, 0]][self.feature_columns].values.astype(np.float32)
        X2_batch = self.data.iloc[batch_pairs[:, 1]][self.feature_columns].values.astype(np.float32)

        if self.augment:
            X1_batch += np.random.normal(0, 0.01, X1_batch.shape)
            X2_batch += np.random.normal(0, 0.01, X2_batch.shape)

        y_contrast = batch_labels.reshape(-1,1).astype('float32')

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
    initializer = tf.keras.initializers.HeNormal()

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
if not os.path.exists(cae_encoder_file) or not os.path.exists(cae_decoder_file):
    print("Training CAE (Dual-Loss) with overrepresented metastatic pairs...")

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
    cae_encoder = tf.keras.models.load_model(cae_encoder_file)
    cae_decoder = tf.keras.models.load_model(cae_decoder_file)


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

# -------------- Evaluate SNN & CAE on Test --------------
print("Evaluating SNN on held-out test set for a direct classification approach...")

# Make sure test_data_scaled was never used in training/validation!
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

# -------------- Generate embeddings & meta-learning --------------
print("Generating embeddings for meta-learning...")

###############################################################################
# 1) Filter out CCLE from train/val/test for meta-learning
###############################################################################
# Make initial copies so that variables are always defined:
train_data_scaled_noCCLE = train_data_scaled.copy()
val_data_scaled_noCCLE   = val_data_scaled.copy()
test_data_scaled_noCCLE  = test_data_scaled.copy()

# Filter out CCLE if it's present:
train_data_scaled_noCCLE = train_data_scaled_noCCLE[train_data_scaled_noCCLE['SOURCE'] != 'CCLE']
val_data_scaled_noCCLE   = val_data_scaled_noCCLE[val_data_scaled_noCCLE['SOURCE'] != 'CCLE']
test_data_scaled_noCCLE  = test_data_scaled_noCCLE[test_data_scaled_noCCLE['SOURCE'] != 'CCLE']

# Now do the “keep only labels that appear in the training portion” step
train_lbls_noCCLE = set(train_data_scaled_noCCLE['LABEL'].unique())
val_data_scaled_noCCLE   = val_data_scaled_noCCLE[val_data_scaled_noCCLE['LABEL'].isin(train_lbls_noCCLE)]
test_data_scaled_noCCLE  = test_data_scaled_noCCLE[test_data_scaled_noCCLE['LABEL'].isin(train_lbls_noCCLE)]

###############################################################################
# 2) Generate SNN + CAE embeddings for train/val/test => shape ~ (None,128)
###############################################################################
snn_base_network = snn_model.layers[2]
# SNN embeddings
snn_emb_train = snn_base_network.predict(train_data_scaled_noCCLE[fcols].values)
snn_emb_val   = snn_base_network.predict(val_data_scaled_noCCLE[fcols].values)
snn_emb_test  = snn_base_network.predict(test_data_scaled_noCCLE[fcols].values)

# CAE embeddings
cae_emb_train = cae_encoder.predict(train_data_scaled_noCCLE[fcols].values)
cae_emb_val   = cae_encoder.predict(val_data_scaled_noCCLE[fcols].values)
cae_emb_test  = cae_encoder.predict(test_data_scaled_noCCLE[fcols].values)

# Combined embeddings => shape (None, 128) if each is 64D
combined_train = np.concatenate([snn_emb_train, cae_emb_train], axis=1)
combined_val   = np.concatenate([snn_emb_val,   cae_emb_val],   axis=1)
combined_test  = np.concatenate([snn_emb_test,  cae_emb_test],  axis=1)

###############################################################################
# 3) Label encoding
###############################################################################
meta_label_encoder = LabelEncoder()
meta_label_encoder.fit(train_data_scaled_noCCLE['LABEL'].unique())

y_met_train = meta_label_encoder.transform(train_data_scaled_noCCLE['LABEL'])
y_met_val   = meta_label_encoder.transform(val_data_scaled_noCCLE['LABEL'])
y_met_test  = meta_label_encoder.transform(test_data_scaled_noCCLE['LABEL'])

met_class_labels = meta_label_encoder.classes_

###############################################################################
# 4) Train & Store Base Classifiers on the EMBEDDINGS (combined_train)
#    so each classifier sees 128-D input => outputs (#labels) prob.
###############################################################################
met_classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                             verbosity=0, objective='multi:softprob')
}
for c_name, clf in met_classifiers.items():
    if c_name == 'XGBoost':
        clf.set_params(num_class=len(met_class_labels))
    clf.fit(combined_train, y_met_train)

# Save base classifiers
with open(os.path.join(evaluation_dir, "trained_base_classifiers.pkl"), "wb") as f:
    pickle.dump(met_classifiers, f)
print("Base classifiers trained & stored (on 128-D embeddings).")

###############################################################################
# 5) Build meta-features from these classifiers => shape (None, n_classifiers * n_labels)
#    We'll apply them to val & test sets of embeddings
###############################################################################
meta_feats_val  = get_meta_features(met_classifiers, combined_val)   # shape e.g. (N_val, 5*#labels)
meta_feats_test = get_meta_features(met_classifiers, combined_test)  # shape e.g. (N_test,5*#labels)

# restricted combos
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
results_cols = ['Iteration','h1','h2','h3','dropout_rate','l2_reg','optimizer',
                'batch_size','epochs','Precision','Recall','ROC_AUC','Accuracy','F1']
results_cols += [f'Accuracy_{lbl}' for lbl in met_class_labels]
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
# 6) Train multiple meta-learner architectures on 'meta_feats_val' => store them
###############################################################################
print("Evaluating restricted meta-learner combos...")
for i, (h1,h2,h3,dr,l2r,opt,bsize,eps) in enumerate(restricted_archs, start=1):
    mm = create_meta_learner(h1=h1, h2=h2, h3=h3,
                             dropout_rate=dr,
                             l2_reg=l2r,
                             optimizer=opt,
                             input_dim=meta_feats_val.shape[1],
                             num_classes=len(met_class_labels))
    mm.fit(meta_feats_val, y_met_val,
           epochs=eps, batch_size=bsize, verbose=1)

    # Store each meta-learner model
    mm_path = os.path.join(evaluation_dir, f"meta_learner_{i}.h5")
    mm.save(mm_path)

    # Evaluate on test set => 'meta_feats_test'
    y_pred_test = mm.predict(meta_feats_test).argmax(axis=1)
    prec = precision_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
    rec  = recall_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
    f1_  = f1_score(y_met_test, y_pred_test, average='weighted', zero_division=0)
    acc  = accuracy_score(y_met_test, y_pred_test)
    y_test_bin = to_categorical(y_met_test, num_classes=len(met_class_labels))
    y_pred_proba = mm.predict(meta_feats_test)
    try:
        roc_ = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except:
        roc_ = np.nan

    cls_accs = get_class_accuracies(y_met_test, y_pred_test, met_class_labels)
    rowd = {
        'Iteration': i,
        'h1': h1, 'h2': h2, 'h3': h3,
        'dropout_rate': dr, 'l2_reg': l2r, 'optimizer': opt,
        'batch_size': bsize, 'epochs': eps,
        'Precision': prec, 'Recall': rec, 'ROC_AUC': roc_,
        'Accuracy': acc, 'F1': f1_
    }
    for cacc_, clbl in zip(cls_accs, [f'Accuracy_{x}' for x in met_class_labels]):
        rowd[clbl] = cacc_
    results_df = results_df.append(rowd, ignore_index=True)

res_csv = os.path.join(evaluation_dir, 'meta_learner_restricted_results.csv')
results_df.to_csv(res_csv, index=False)
print(f"Meta-learner results saved => {res_csv}")

###############################################################################
# 7) Evaluate the top-2 meta-learners on subsets (TCGA+GTEX vs. MET)
#    using the same SNN+CAE embeddings => base classifiers => meta-features
###############################################################################
print("Preparing subsets for Confusion Matrices of the best meta-learners...")

# A) Subset test_data_scaled_noCCLE
tcga_gtex_test = test_data_scaled_noCCLE[test_data_scaled_noCCLE['SOURCE'].isin(['TCGA','GTEX'])].copy()
met_test       = test_data_scaled_noCCLE[test_data_scaled_noCCLE['SOURCE'] == 'METASTATIC'].copy()

# B) True labels (numeric)
y_true_tcga_gtex = meta_label_encoder.transform(tcga_gtex_test['LABEL'])
y_true_met       = meta_label_encoder.transform(met_test['LABEL'])

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
meta_feats_tg_test  = get_meta_features(met_classifiers, X_tg_test_emb)
meta_feats_met_test = get_meta_features(met_classifiers, X_met_test_emb)

# E) Sort and pick top-2 by accuracy
results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False)
best_two_models   = results_df_sorted.head(2).copy()

tg_classes  = meta_label_encoder.classes_
met_classes = meta_label_encoder.classes_

for idx, row in best_two_models.iterrows():
    iteration = int(row['Iteration'])
    h1 = row['h1']
    h2 = row['h2']
    h3 = row['h3']
    dr = row['dropout_rate']
    l2r = row['l2_reg']
    opt = row['optimizer']
    bsize = int(row['batch_size'])
    eps = int(row['epochs'])

    print(f"\nRe-initializing meta-learner from iteration={iteration} with highest Accuracy={row['Accuracy']:.3f}")

    # F) Rebuild & re-train the same meta-learner on meta_feats_val
    meta_model = create_meta_learner(
        h1=h1, h2=h2, h3=h3,
        dropout_rate=dr, l2_reg=l2r, optimizer=opt,
        input_dim=meta_feats_val.shape[1],
        num_classes=len(met_class_labels)
    )
    meta_model.fit(meta_feats_val, y_met_val,
                   epochs=eps, batch_size=bsize, verbose=0)

    # Also store these top-2 meta models
    best_path = os.path.join(evaluation_dir, f"best_meta_learner_{iteration}.h5")
    meta_model.save(best_path)

    # G) Predict integer labels on meta-features of each subset
    y_pred_tcga_gtex = meta_model.predict(meta_feats_tg_test).argmax(axis=1)
    y_pred_met       = meta_model.predict(meta_feats_met_test).argmax(axis=1)

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
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get all non-met labels (you may sort them alphabetically)
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

# We'll pick test_data for: metastatic only
test_data_scaled_noCCLE = test_data_scaled[test_data_scaled['LABEL'].isin(train_lbls_noCCLE)]
met_only = test_data_scaled_noCCLE[test_data_scaled_noCCLE['SOURCE'] == 'METASTATIC']
tcga_gtex_ccle = test_data_scaled[test_data_scaled['SOURCE'].isin(['TCGA', 'GTEX', 'CCLE'])]

X_met_pre = met_only[fcols].values
X_met_post_snn = snn_base_network.predict(met_only[fcols].values)
X_met_post_cae = cae_encoder.predict(met_only[fcols].values)
X_met_post = np.concatenate([X_met_post_snn, X_met_post_cae], axis=1)
met_labels = met_only['LABEL'].values
unique_met_labels = np.unique(met_labels)

X_tgc_pre = tcga_gtex_ccle[fcols].values
X_tgc_post_snn = snn_base_network.predict(tcga_gtex_ccle[fcols].values)
X_tgc_post_cae = cae_encoder.predict(tcga_gtex_ccle[fcols].values)
X_tgc_post = np.concatenate([X_tgc_post_snn, X_tgc_post_cae], axis=1)
tgc_labels = tcga_gtex_ccle['LABEL'].values
unique_tgc_labels = np.unique(tgc_labels)


#############################
# Modified 2D Dimension Reduction Plot Function
#############################
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


#############################
# Modified 3D Dimension Reduction Plot Function
#############################
def plot_dim_reductions_3d(X_pre, X_post, labels, set_name, output_dir, unique_labels, hide_legend=False):
    """
    Creates a 3D dimension reduction figure using t-SNE, PCA, and UMAP.
    In each subplot the top-right annotation shows only the median distortion.
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
                       "TCGA_GTEX_CCLE", evaluation_dir, unique_tgc_labels,
                       hide_legend=True)

plot_dim_reductions_3d(X_met_pre, X_met_post, met_labels,
                       "MetastaticOnly", evaluation_dir, unique_met_labels,
                       hide_legend=False)
plot_dim_reductions_3d(X_tgc_pre, X_tgc_post, tgc_labels,
                       "TCGA_GTEX_CCLE", evaluation_dir, unique_tgc_labels,
                       hide_legend=True)

###############################################################################
# 3D PCA & 3D tSNE 4-Axis Plots (Non-Met vs. Met, Pre vs. Post), with Cluster Centroids
###############################################################################
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from sklearn.metrics import silhouette_samples  # (Not used anymore for display)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
# NEW: Plot ALL Samples (including metastatic tissue) using PCA (2D and 3D) – Only Distortion
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

    all_labels    = test_df["LABEL"].values
    unique_labels = np.unique(all_labels)

    # consistent colours
    palette   = sns.color_palette("hls", len(unique_labels))
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
    pre_2d  = pca2.fit_transform(X_all_pre)
    post_2d = pca2.fit_transform(X_all_post)

    pca3 = PCA(n_components=3, random_state=42)
    pre_3d  = pca3.fit_transform(X_all_pre)
    post_3d = pca3.fit_transform(X_all_post)

    # ---------- figure ----------
    fig = plt.figure(figsize=(12, 10))

    # 2-D
    ax00 = plt.subplot2grid((2, 2), (0, 0))
    ax01 = plt.subplot2grid((2, 2), (0, 1))
    _scatter_2d(ax00, pre_2d,  all_labels, "ALL Samples Pre (2D PCA)")
    _scatter_2d(ax01, post_2d, all_labels, "ALL Samples Post (2D PCA)")

    # 3-D
    ax10 = plt.subplot2grid((2, 2), (1, 0), projection="3d")
    ax11 = plt.subplot2grid((2, 2), (1, 1), projection="3d")
    _scatter_3d(ax10, pre_3d,  all_labels, "ALL Samples Pre (3D PCA)")
    _scatter_3d(ax11, post_3d, all_labels, "ALL Samples Post (3D PCA)")


    fig.suptitle("PCA of ALL Samples (Pre vs. Post)",
                 fontsize=mpl.rcParams["figure.titlesize"])
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

###############################################################################
# NEW: Plot ALL Samples (including metastatic tissue) using PCA (2D and 3D) – Distortion & Silhouette
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
# Another Stage: Compare SNN-only vs CAE-only vs Combined Meta-Learners
# with a bar plot from 0.75 to 1, legend outside on the right,
# Blue=Accuracy, Red=F1
###############################################################################
print("\n--- Comparing SNN, CAE, and Combined embeddings via meta-learner ---")

# 1) Retrieve "best" meta-learner architecture from earlier
best_model_row = best_two_models.iloc[0]  # pick the single best row
h1  = int(best_model_row['h1'])
h2  = int(best_model_row['h2'])
h3  = int(best_model_row['h3'])
dr  = float(best_model_row['dropout_rate'])
l2r = float(best_model_row['l2_reg'])
opt = best_model_row['optimizer']
bsize= int(best_model_row['batch_size'])
eps  = int(best_model_row['epochs'])

# 2) We'll train a meta-learner on SNN embeddings only
print("Training meta-learner on SNN embeddings alone...")
mm_snn = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=snn_emb_train.shape[1],  # only SNN dims
    num_classes=len(met_class_labels)
)
mm_snn.fit(snn_emb_val, y_met_val, epochs=eps, batch_size=bsize, verbose=0)
y_pred_snn_test = mm_snn.predict(snn_emb_test).argmax(axis=1)

acc_snn = accuracy_score(y_met_test, y_pred_snn_test)
f1_snn  = f1_score(y_met_test, y_pred_snn_test, average='weighted', zero_division=0)

print(f"SNN-Only => Accuracy={acc_snn:.3f}, F1={f1_snn:.3f}")

# 3) Train a meta-learner on CAE embeddings only
print("Training meta-learner on CAE embeddings alone...")
mm_cae = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=cae_emb_train.shape[1],  # only CAE dims
    num_classes=len(met_class_labels)
)
mm_cae.fit(cae_emb_val, y_met_val, epochs=eps, batch_size=bsize, verbose=0)
y_pred_cae_test = mm_cae.predict(cae_emb_test).argmax(axis=1)

acc_cae = accuracy_score(y_met_test, y_pred_cae_test)
f1_cae  = f1_score(y_met_test, y_pred_cae_test, average='weighted', zero_division=0)

print(f"CAE-Only => Accuracy={acc_cae:.3f}, F1={f1_cae:.3f}")

# 4) "Combined" => use best meta-learner's final accuracy & F1 from earlier
acc_comb = float(best_model_row['Accuracy'])
f1_comb  = float(best_model_row['F1'])

print(f"Combined => Accuracy={acc_comb:.3f}, F1={f1_comb:.3f}")

# 5) Make a grouped bar plot: x-axis => [SNN, CAE, Combined]
#    We'll have 2 bars per group => Accuracy, F1
embedding_labels = ["SNN", "CAE", "Combined"]
acc_values = [acc_snn, acc_cae, acc_comb]
f1_values  = [f1_snn,  f1_cae,  f1_comb]

x = np.arange(len(embedding_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6,5))

# Accuracy bars (blue)
bar_acc = ax.bar(x - width/2, acc_values, width=width,
                 label="Accuracy", color="blue")
# F1 bars (red)
bar_f1  = ax.bar(x + width/2, f1_values, width=width,
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
                xy=(rect.get_x() + rect.get_width()/2, height),
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
# in 2D or 3D. No on-graph label annotation; median distortion goes into legend.
# The legend is fully below the figure in multiple columns to avoid overlap.
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
        palette   = sns.color_palette("hls", len(unique_labels))
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

    _plot(ax00, _transform(X_pre_nonmet),  labels_pre_nonmet,  "Non-Met PRE")
    _plot(ax01, _transform(X_post_nonmet), labels_post_nonmet, "Non-Met POST")
    _plot(ax10, _transform(X_pre_met),     labels_pre_met,     "Met PRE")
    _plot(ax11, _transform(X_post_met),    labels_post_met,    "Met POST")

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

print(f"Using meta-learner from iteration={best_iteration} with Accuracy={best_model_row['Accuracy']:.4f}")

# 2) Rebuild & re-train that top meta-learner on meta_feats_val
meta_model = create_meta_learner(
    h1=h1, h2=h2, h3=h3,
    dropout_rate=dr, l2_reg=l2r, optimizer=opt,
    input_dim=meta_feats_val.shape[1],
    num_classes=len(met_class_labels)
)
meta_model.fit(meta_feats_val, y_met_val, epochs=eps, batch_size=bsize, verbose=0)

# 3) Compute baseline accuracy on the original test set (meta_feats_test)
y_pred_test_baseline = meta_model.predict(meta_feats_test).argmax(axis=1)
acc_baseline = accuracy_score(y_met_test, y_pred_test_baseline)
print(f"Baseline test accuracy (no noise) = {acc_baseline:.4f}")

# -------------------------------------------------------------------------
# 3a) Additionally, compute baseline accuracy *per label*, e.g., "Brain_GTEX",
#     "PAAD_METASTATIC", etc.
# -------------------------------------------------------------------------
test_data_noCCLE_df = test_data_scaled_noCCLE.copy().reset_index(drop=True)

# We'll produce baseline predictions on this test set for each label
test_X_snn = snn_base_network.predict(test_data_noCCLE_df[fcols].values)
test_X_cae = cae_encoder.predict(test_data_noCCLE_df[fcols].values)
test_combined_emb = np.concatenate([test_X_snn, test_X_cae], axis=1)
test_meta_feats = get_meta_features(met_classifiers, test_combined_emb)
test_y_pred = meta_model.predict(test_meta_feats).argmax(axis=1)

unique_labels_for_acc = test_data_noCCLE_df['LABEL'].unique()
baseline_label_acc = {}
for lbl in unique_labels_for_acc:
    idx_ = (test_data_noCCLE_df['LABEL'] == lbl)
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
        test_noisy_df = test_data_noCCLE_df.copy()

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
