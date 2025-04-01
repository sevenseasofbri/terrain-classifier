import os
import glob
import numpy as np
import librosa # used only for loading audio files

# -------------------------------
# Helper: Framing the audio signal
# -------------------------------
def frame_signal(y, frame_length, hop_length):
    """Split the signal into overlapping frames."""
    n_samples = len(y)
    if n_samples < frame_length:
        return np.empty((0, frame_length))
    n_frames = 1 + (n_samples - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

# -------------------------------
# Time-Domain Feature Functions
# -------------------------------
def zcr(frame, frame_length):
    """Compute Zero Crossing Rate (ZCR) using numpy."""
    # Count sign changes and normalize by frame length.
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_length)

def rms(frame):
    """Compute Root Mean Square (RMS) value."""
    return np.sqrt(np.mean(frame ** 2))

def temporal_entropy(frame):
    """Compute Temporal Entropy of a frame."""
    hist, _ = np.histogram(frame, bins=8, range=(np.min(frame), np.max(frame)))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # Avoid log(0)
    return -np.sum(prob * np.log2(prob))

# -------------------------------
# Frequency-Domain Feature Functions
# -------------------------------
def compute_fft(y, frame_length, hop_length):
    """
    Compute the Short-Time Fourier Transform (STFT) using numpy.
    Returns the magnitude spectrogram.
    """
    frames = frame_signal(y, frame_length, hop_length)
    if frames.size == 0:
        return np.empty((0, frame_length // 2 + 1))
    window = np.hanning(frame_length)
    frames_win = frames * window
    fft_frames = np.fft.rfft(frames_win, axis=1)
    return np.abs(fft_frames)

def spectral_centroid(S, sr):
    """Compute the spectral centroid for each frame."""
    if S.shape[0] == 0:
        return np.array([])
    n_freqs = S.shape[1]
    freqs = np.linspace(0, sr / 2, n_freqs)
    centroids = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        centroids[i] = np.sum(freqs * S[i]) / (np.sum(S[i]) + 1e-10)
    return centroids

def spectral_rolloff(S, sr, roll_percent=0.85):
    """Compute the spectral rolloff for each frame."""
    if S.shape[0] == 0:
        return np.array([])
    n_freqs = S.shape[1]
    freqs = np.linspace(0, sr / 2, n_freqs)
    rolloffs = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        total_energy = np.sum(S[i])
        threshold = roll_percent * total_energy
        cumulative_energy = np.cumsum(S[i])
        idx = np.where(cumulative_energy >= threshold)[0]
        if idx.size > 0:
            rolloffs[i] = freqs[idx[0]]
        else:
            rolloffs[i] = freqs[-1]
    return rolloffs

def spectral_flatness(S):
    """Compute the spectral flatness for each frame."""
    if S.shape[0] == 0:
        return np.array([])
    flatness = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        gm = np.exp(np.mean(np.log(S[i] + 1e-10)))
        am = np.mean(S[i])
        flatness[i] = gm / (am + 1e-10)
    return flatness

def band_ratio(S, sr, frame_length):
    """Compute the band energy ratio (mid vs. low frequencies) for each frame."""
    if S.shape[0] == 0:
        return np.array([])
    n_freqs = S.shape[1]
    freqs = np.linspace(0, sr / 2, n_freqs)
    ratios = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        low_energy = np.sum(S[i][(freqs >= 100) & (freqs < 1000)])
        mid_energy = np.sum(S[i][(freqs >= 1000) & (freqs < 4000)])
        ratios[i] = mid_energy / (low_energy + 1e-10)
    return ratios

# -------------------------------
# Audio Feature Extraction
# -------------------------------
def extract_audio_features(file_path, sr, frame_length=2048, hop_length=512):
    """
    Load an audio file and extract a vector of features.
    Uses librosa only for loading the audio.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=5.0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    if y is None or len(y) == 0:
        print(f"Empty audio file: {file_path}")
        return None

    # Frame the signal for time-domain feature extraction.
    frames = frame_signal(y, frame_length, hop_length)
    if frames.size == 0:
        print(f"Audio too short for framing: {file_path}")
        return None
    n_frames = frames.shape[0]

    # Compute time-domain features per frame.
    time_features = {
        'zcr': np.empty(n_frames),
        'rms': np.empty(n_frames),
        'temporal_entropy': np.empty(n_frames)
    }
    for i in range(n_frames):
        frame = frames[i]
        time_features['zcr'][i] = zcr(frame, frame_length)
        time_features['rms'][i] = rms(frame)
        time_features['temporal_entropy'][i] = temporal_entropy(frame)

    # Compute frequency-domain features using our numpy STFT.
    S = compute_fft(y, frame_length, hop_length)
    if S.size == 0:
        print(f"STFT computation failed for: {file_path}")
        return None
    freq_features = {
        'spectral_centroid': spectral_centroid(S, sr),
        'spectral_rolloff': spectral_rolloff(S, sr),
        'spectral_flatness': spectral_flatness(S),
        'band_ratio': band_ratio(S, sr, frame_length)
    }

    # Aggregate statistics (mean, std, max, etc.) for each feature.
    feature_vector = [
        np.mean(time_features['zcr']), np.std(time_features['zcr']),
        np.mean(time_features['rms']), np.max(time_features['rms']),
        np.mean(time_features['temporal_entropy']),
        np.mean(freq_features['spectral_centroid']),
        np.mean(freq_features['spectral_rolloff']),
        np.mean(freq_features['spectral_flatness']),
        np.mean(freq_features['band_ratio'])
    ]
    return feature_vector

# -------------------------------
# Dataset Loading Function
# -------------------------------
def load_dataset(dataset_type="train"):
    """Load dataset file paths organized in class subdirectories."""
    audio_folder = os.path.join("audio", dataset_type)
    classes = sorted(os.listdir(audio_folder))
    audio_files = {}
    for c in classes:
        audio_files[c] = sorted(glob.glob(os.path.join(audio_folder, c, "*.wav")))
    print("Classes:", classes)
    return audio_files, classes

# -------------------------------
# Hyperdimensional Computing Functions
# -------------------------------
feature_names = [
    'zcr_mean', 'zcr_std', 'rms_mean', 'rms_max',
    'entropy_mean', 'spectral_centroid_mean', 'spectral_rolloff_mean',
    'spectral_flatness_mean', 'band_ratio_mean'
]

important_pairs = [
    ('zcr_mean', 'entropy_mean'),
    ('rms_mean', 'spectral_rolloff_mean'),
    ('spectral_flatness_mean', 'spectral_centroid_mean'),
    ('rms_max', 'band_ratio_mean')
]

def generate_feature_codebook(feature_names, D):
    """Generate a codebook for feature names."""
    base = np.random.randint(0, 2, D, dtype=np.uint8)
    codebook = {}
    for i in range(len(feature_names)):
        # Shift the base vector by i+1 positions
        codebook[feature_names[i]] = np.roll(base, i + 1)
    return codebook

def generate_value_level_hvs(levels, D):
    """Pre-generate value level hypervectors."""
    level_hvs = []
    for level in range(levels):
        hv = np.zeros(D, dtype=np.uint8)
        if level > 0:
            n_bits = level * D // levels
            indices = np.random.choice(D, n_bits, replace=False)
            hv[indices] = 1
        level_hvs.append(hv)
    return level_hvs

def get_value_hv(levels, value, level_hvs):
    """Map a normalized value (0-1) to the nearest level hypervector."""
    level = int(value * levels)
    if level >= levels:
        level = levels - 1
    if level < 0:
        level = 0
    return level_hvs[level]

def encode_feature_vector(features, codebook, level_hvs, D, levels):
    """Encode a single feature vector into a hypervector."""
    # Build a feature dictionary by mapping feature names to values.
    feature_dict = {}
    for i in range(len(codebook)):
        key = list(codebook.keys())[i]
        feature_dict[key] = features[i]

    hvs = []
    # Encode individual features.
    for name in feature_dict:
        feat_hv = np.bitwise_xor(codebook[name], get_value_hv(levels, feature_dict[name], level_hvs))
        hvs.append(feat_hv.astype(np.int16))

    # Encode selected feature-pair interactions.
    for pair in important_pairs:
        f1, f2 = pair
        hv1 = np.bitwise_xor(codebook[f1], get_value_hv(levels, feature_dict[f1], level_hvs))
        hv2 = np.bitwise_xor(codebook[f2], get_value_hv(levels, feature_dict[f2], level_hvs))
        pair_hv = np.bitwise_xor(hv1, hv2)
        hvs.append(pair_hv.astype(np.int16))

    # Bundle the hypervectors using majority vote.
    hvs_array = np.array(hvs)
    sum_hv = np.sum(hvs_array, axis=0)
    threshold = len(hvs) // 2
    final_hv = (sum_hv > threshold).astype(np.uint8)
    return final_hv

def train_hd_classifier(dataset, labels, epochs, D, levels):
    """
    Train a hyperdimensional classifier.
    dataset: array of encoded hypervectors (N x D)
    labels: corresponding class labels (N)
    """
    num_classes = int(np.max(labels)) + 1
    real_class_hvs = np.zeros((num_classes, D), dtype=np.int16)
    N = len(dataset)

    for epoch in range(epochs):
        for i in range(N):
            query_hv = dataset[i]
            y_true = labels[i]
            # Binarize class hypervectors.
            bin_class_hvs = (real_class_hvs >= 0).astype(np.uint8)
            # Predict label by computing Hamming distances.
            distances = []
            for j in range(num_classes):
                distances.append(np.sum(query_hv != bin_class_hvs[j]))
            y_pred = np.argmin(distances)
            # Update if prediction is wrong.
            if y_pred != y_true:
                real_class_hvs[y_true] += query_hv
                real_class_hvs[y_pred] -= query_hv
        # Shuffle the training data.
        perm = np.random.permutation(N)
        dataset = dataset[perm]
        labels = labels[perm]
    final_class_hvs = (real_class_hvs >= 0).astype(np.uint8)
    return final_class_hvs

def predict_hd(query_hv, class_hvs):
    """Predict the class label for a query hypervector."""
    num_classes = class_hvs.shape[0]
    distances = np.empty(num_classes)
    for j in range(num_classes):
        distances[j] = np.sum(query_hv != class_hvs[j])
    return int(np.argmin(distances))

def evaluate_hd(vectors, labels, class_hvs):
    """Evaluate the classifier's accuracy without using zip."""
    correct = 0
    N = len(vectors)
    for i in range(N):
        query_hv = vectors[i]
        y_true = labels[i]
        y_pred = predict_hd(query_hv, class_hvs)
        if y_pred == y_true:
            correct += 1
    return correct / N

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Settings
    sr = 4000            # Sampling rate for audio loading
    frame_length = 2048  # Frame length for analysis
    hop_length = 512     # Hop length for framing
    D = 10000            # Dimensionality of hypervectors
    LEVELS = 256         # Quantization levels
    epochs = 10          # Training epochs

    np.random.seed(42)

    # ----- Training Phase -----
    print("=== Training Phase ===")
    train_dataset = []
    train_labels = []
    audio_files, classes = load_dataset("train")
    label_to_id = {}
    for idx in range(len(classes)):
        label_to_id[classes[idx]] = idx

    # Extract features for each training audio file.
    for cls in classes:
        for file_path in audio_files[cls]:
            features = extract_audio_features(file_path, sr, frame_length, hop_length)
            if features is not None:
                train_dataset.append(features)
                train_labels.append(label_to_id[cls])
    if len(train_dataset) == 0:
        print("No training data found.")
        return

    train_dataset = np.array(train_dataset)
    # Normalize features to [0, 1]
    min_vals = train_dataset.min(axis=0)
    max_vals = train_dataset.max(axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1e-10
    train_dataset = (train_dataset - min_vals) / range_vals

    # Generate codebook and value-level hypervectors.
    codebook = generate_feature_codebook(feature_names, D)
    value_level_hvs = generate_value_level_hvs(LEVELS, D)

    # Encode training feature vectors.
    train_vectors = []
    for i in range(len(train_dataset)):
        encoded = encode_feature_vector(train_dataset[i], codebook, value_level_hvs, D, LEVELS)
        train_vectors.append(encoded)
    train_vectors = np.array(train_vectors)
    train_labels = np.array(train_labels)

    # Train the classifier.
    class_hvs = train_hd_classifier(train_vectors, train_labels, epochs, D, LEVELS)
    train_acc = evaluate_hd(train_vectors, train_labels, class_hvs)
    print(f"Training accuracy: {train_acc * 100:.2f}%")

    # ----- Testing Phase -----
    print("\n=== Testing Phase ===")
    test_dataset = []
    test_labels = []
    audio_files, classes = load_dataset("test")
    # Use the same label mapping
    for cls in classes:
        for file_path in audio_files[cls]:
            features = extract_audio_features(file_path, sr, frame_length, hop_length)
            if features is not None:
                test_dataset.append(features)
                test_labels.append(label_to_id.get(cls, -1))
    if len(test_dataset) == 0:
        print("No testing data found.")
        return

    test_dataset = np.array(test_dataset)
    # Normalize features using the training set range.
    test_dataset = (test_dataset - min_vals) / range_vals

    test_vectors = []
    for i in range(len(test_dataset)):
        encoded = encode_feature_vector(test_dataset[i], codebook, value_level_hvs, D, LEVELS)
        test_vectors.append(encoded)
    test_vectors = np.array(test_vectors)
    test_labels = np.array(test_labels)

    test_acc = evaluate_hd(test_vectors, test_labels, class_hvs)
    print(f"Testing accuracy: {test_acc * 100:.2f}%")

if __name__ == '__main__':
    main()
