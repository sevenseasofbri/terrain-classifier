#include <stdio.h>
// #include "Include/arm_math.h"   // CMSIS DSP header
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>    // Complex number operations
#include <stdbool.h>
#include "weights.h"    // Preloaded hyperdimensional components
#include "audio_data.h"  // Preloaded audio data
// ==========================
// Configuration Parameters
// ==========================
#define FRAME_LENGTH    2048
#define HOP_LENGTH      512
#define SAMPLE_RATE     8000
#define FFT_SIZE        FRAME_LENGTH
#define NUM_BINS        (FFT_SIZE/2 + 1)
#define NUM_HIST_BINS   8

// Hyperdimensional settings - defined in the weights.h file
// #define D               
// #define NUM_FEATURES    
// #define LEVELS          
// #define NUM_CLASSES     

// Important feature pairs (indices in the feature vector)
// Feature order assumed:
//   0: zcr_mean, 1: zcr_std, 2: rms_mean, 3: rms_max,
//   4: entropy_mean, 5: spectral_centroid_mean, 6: spectral_rolloff_mean,
//   7: spectral_flatness_mean, 8: band_ratio_mean
typedef struct {
    int f1;
    int f2;
} Pair;
static Pair important_pairs[] = {
    {0, 4},  // (zcr_mean, entropy_mean)
    {2, 6},  // (rms_mean, spectral_rolloff_mean)
    {7, 5},  // (spectral_flatness_mean, spectral_centroid_mean)
    {3, 8}   // (rms_max, band_ratio_mean)
};
#define NUM_PAIRS (sizeof(important_pairs) / sizeof(important_pairs[0]))

// =====================================================
// Preloaded Hyperdimensional Components (load them from the weights.h file)
// =====================================================


// Normalization parameters (from training) for each feature.
static float feature_min[NUM_FEATURES]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static float feature_range[NUM_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// ==========================
// Function Prototypes
// ==========================
int load_audio_file(const char *filename, float **audio_data, int *length);
void extract_features(const float *audio_data, int length, float *feature_vector);

float compute_zcr(const float *frame, int length);
float compute_rms(const float *frame, int length);
float compute_temporal_entropy(const float *frame, int length);

void compute_fft(const float *frame, float *mag, int fft_size);
float compute_spectral_centroid(const float *mag, int num_bins, float sample_rate, int fft_size);
float compute_spectral_rolloff(const float *mag, int num_bins, float sample_rate, int fft_size, float roll_percent);
float compute_spectral_flatness(const float *mag, int num_bins);
float compute_band_ratio(const float *mag, int num_bins, float sample_rate, int fft_size);

int quantize(float value, int levels);
void encode_feature_vector(const float *features, bool *hv);
// void encode_feature_vector(const float *features, uint8_t *hv);
// int hamming_distance(const uint8_t *hv1, const uint8_t *hv2, int d);
int hamming_distance(bool *hv1, const bool *hv2, int d);


// ==========================
// Feature Extraction Functions
// ==========================

// Compute Zero Crossing Rate (ZCR) for a frame.
float compute_zcr(const float *frame, int length) {
    // print the frame
    // for (int i = 0; i < length; i++) {
    //     printf("%f ", frame[i]);
    // }
    int count = 0;
    for (int i = 1; i < length; i++) {
        if ((frame[i - 1] >= 0 && frame[i] < 0) ||
            (frame[i - 1] < 0 && frame[i] >= 0)) {
            count++;
        }
    }
    return (float)count / (float)length;
}

// Compute Root Mean Square (RMS) for a frame.
float compute_rms(const float *frame, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += frame[i] * frame[i];
    }
    return sqrt(sum / length);
}

// Compute Temporal Entropy for a frame using NUM_HIST_BINS bins.
float compute_temporal_entropy(const float *frame, int length) {
    float min_val = frame[0], max_val = frame[0];
    for (int i = 1; i < length; i++) {
        if (frame[i] < min_val) min_val = frame[i];
        if (frame[i] > max_val) max_val = frame[i];
    }
    float range = max_val - min_val;
    if (range <= 0.0f) range = 1e-6f;
    float bin_width = range / NUM_HIST_BINS;
    int histogram[NUM_HIST_BINS] = {0};
    for (int i = 0; i < length; i++) {
        int bin = (int)((frame[i] - min_val) / bin_width);
        if (bin >= NUM_HIST_BINS) bin = NUM_HIST_BINS - 1;
        histogram[bin]++;
    }
    float entropy = 0.0f;
    for (int i = 0; i < NUM_HIST_BINS; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / length;
            entropy -= p * (log(p) / log(2.0f));
        }
    }
    return entropy;
}

#define PI 3.14159265358979323846

// Complex number struct
typedef struct {
    float re;
    float im;
} Complex;

// Bit reversal utility
static unsigned int reverse_bits(unsigned int x, unsigned int n) {
    unsigned int result = 0;
    for (unsigned int i = 0; i < n; i++) {
        if ((x >> i) & 1) {
            result |= 1 << (n - 1 - i);
        }
    }
    return result;
}

// In-place Cooley-Tukey FFT
void fft(Complex *buf, int n) {
    int levels = log2(n);
    for (int i = 0; i < n; i++) {
        int j = reverse_bits(i, levels);
        if (j > i) {
            Complex tmp = buf[i];
            buf[i] = buf[j];
            buf[j] = tmp;
        }
    }

    for (int size = 2; size <= n; size *= 2) {
        float angle = -2.0 * PI / size;
        Complex w_m = { cos(angle), sin(angle) };
        for (int start = 0; start < n; start += size) {
            Complex w = { 1.0, 0.0 };
            for (int j = 0; j < size / 2; j++) {
                Complex t, u;
                t.re = w.re * buf[start + j + size / 2].re - w.im * buf[start + j + size / 2].im;
                t.im = w.re * buf[start + j + size / 2].im + w.im * buf[start + j + size / 2].re;
                u = buf[start + j];

                buf[start + j].re = u.re + t.re;
                buf[start + j].im = u.im + t.im;

                buf[start + j + size / 2].re = u.re - t.re;
                buf[start + j + size / 2].im = u.im - t.im;

                float tmp_re = w.re * w_m.re - w.im * w_m.im;
                w.im = w.re * w_m.im + w.im * w_m.re;
                w.re = tmp_re;
            }
        }
    }
}

// Main API: Compute magnitude spectrum of a windowed real frame
void compute_fft(const float *frame, float *mag, int fft_size) {
    Complex *x = (Complex *)malloc(sizeof(Complex) * fft_size);
    if (!x) return;

    // Apply Hann window and convert to complex
    for (int i = 0; i < fft_size; i++) {
        float w = 0.5f - 0.5f * cosf(2 * PI * i / (fft_size - 1));
        x[i].re = frame[i] * w;
        x[i].im = 0.0f;
    }

    // Perform FFT
    fft(x, fft_size);

    // Compute magnitude (only first N/2 + 1 bins, like librosa)
    int bins = fft_size / 2 + 1;
    for (int i = 0; i < bins; i++) {
        mag[i] = sqrtf(x[i].re * x[i].re + x[i].im * x[i].im);
    }

    free(x);
}

// Compute Spectral Centroid.
float compute_spectral_centroid(const float *mag, int num_bins, float sample_rate, int fft_size) {
    float bin_width = sample_rate / (float)fft_size;
    float weighted_sum = 0.0f, sum_mag = 0.0f;
    for (int k = 0; k < num_bins; k++) {
        weighted_sum += k * bin_width * mag[k];
        sum_mag += mag[k];
    }
    return weighted_sum / (sum_mag + 1e-10f);
}

// Compute Spectral Rolloff.
float compute_spectral_rolloff(const float *mag, int num_bins, float sample_rate, int fft_size, float roll_percent) {
    float total_energy = 0.0f;
    for (int k = 0; k < num_bins; k++) {
        total_energy += mag[k];
    }
    float threshold = roll_percent * total_energy;
    float cumulative = 0.0f;
    float bin_width = sample_rate / (float)fft_size;
    for (int k = 0; k < num_bins; k++) {
        cumulative += mag[k];
        if (cumulative >= threshold) {
            return k * bin_width;
        }
    }
    return (num_bins - 1) * bin_width;
}

// Compute Spectral Flatness.

float compute_spectral_flatness(const float *mag, int num_bins) {
    float log_sum = 0.0f;
    float arith_sum = 0.0f;
    for (int k = 0; k < num_bins; k++) {
        float x = mag[k] + 1e-10f;  // avoid log(0) or divide-by-zero
        log_sum += logf(x);
        arith_sum += x;
    }
    float geo_mean = expf(log_sum / num_bins);
    float arith_mean = arith_sum / num_bins;
    return geo_mean / arith_mean;
}


// Compute Band Ratio: energy in 1000-4000 Hz vs. 100-1000 Hz.
float compute_band_ratio(const float *mag, int num_bins, float sample_rate, int fft_size) {
    float bin_width = sample_rate / (float)fft_size;
    float low_energy = 0.0f, mid_energy = 0.0f;
    for (int k = 0; k < num_bins; k++) {
        float freq = k * bin_width;
        if (freq >= 100.0f && freq < 1000.0f)
            low_energy += mag[k];
        else if (freq >= 1000.0f && freq < 4000.0f)
            mid_energy += mag[k];
    }
    return mid_energy / (low_energy + 1e-10f);
}

// ==========================
// Aggregated Feature Extraction
// ==========================
// This function divides the audio into overlapping frames, computes per-frame features,
// and then aggregates them (e.g., mean, std, max) into a single feature vector.
void extract_features(const float *audio_data, int length, float *feature_vector) {
    int num_frames = 0;
    if (length >= FRAME_LENGTH)
        num_frames = 1 + (length - FRAME_LENGTH) / HOP_LENGTH;
    else {
        printf("Audio too short for processing.\n");
        return;
    }

    // Allocate arrays for per-frame features.
    float *zcr_vals       = (float *)malloc(sizeof(float) * num_frames);
    float *rms_vals       = (float *)malloc(sizeof(float) * num_frames);
    float *entropy_vals   = (float *)malloc(sizeof(float) * num_frames);
    float *centroid_vals  = (float *)malloc(sizeof(float) * num_frames);
    float *rolloff_vals   = (float *)malloc(sizeof(float) * num_frames);
    float *flatness_vals  = (float *)malloc(sizeof(float) * num_frames);
    float *band_ratio_vals= (float *)malloc(sizeof(float) * num_frames);
    float *mag = (float *)malloc(sizeof(float) * NUM_BINS);

    if (!zcr_vals || !rms_vals || !entropy_vals || !centroid_vals ||
        !rolloff_vals || !flatness_vals || !band_ratio_vals || !mag) {
        printf("Memory allocation failed for feature arrays.\n");
        free(zcr_vals); free(rms_vals); free(entropy_vals); free(centroid_vals);
        free(rolloff_vals); free(flatness_vals); free(band_ratio_vals); free(mag);
        return;
    }
    // printf("Memory allocated for feature arrays.\n");
    // Process each frame.
    for (int i = 0; i < num_frames; i++) {
        int start = i * HOP_LENGTH;
        const float *frame = audio_data + start;
        zcr_vals[i] = compute_zcr(frame, FRAME_LENGTH);
        printf("Frame %d: %f\n", i, zcr_vals[i]);
        rms_vals[i] = compute_rms(frame, FRAME_LENGTH);
        // printf("RMS %f\n", rms_vals[i]);
        entropy_vals[i] = compute_temporal_entropy(frame, FRAME_LENGTH);
        // printf("Good till here %d\n", i);
        compute_fft(frame, mag, FFT_SIZE);
        centroid_vals[i] = compute_spectral_centroid(mag, NUM_BINS, SAMPLE_RATE, FFT_SIZE);
        rolloff_vals[i] = compute_spectral_rolloff(mag, NUM_BINS, SAMPLE_RATE, FFT_SIZE, 0.85f);
        flatness_vals[i] = compute_spectral_flatness(mag, NUM_BINS);
        band_ratio_vals[i] = compute_band_ratio(mag, NUM_BINS, SAMPLE_RATE, FFT_SIZE);
    }

    // Aggregate statistics
    float sum_zcr = 0.0f, sum_zcr_sq = 0.0f;
    float sum_rms = 0.0f, max_rms = 0.0f;
    float sum_entropy = 0.0f;
    float sum_centroid = 0.0f, sum_rolloff = 0.0f;
    float sum_flatness = 0.0f, sum_band_ratio = 0.0f;

    for (int i = 0; i < num_frames; i++) {
        sum_zcr += zcr_vals[i];
        sum_zcr_sq += zcr_vals[i] * zcr_vals[i];
        sum_rms += rms_vals[i];
        if (rms_vals[i] > max_rms) max_rms = rms_vals[i];
        sum_entropy += entropy_vals[i];
        sum_centroid += centroid_vals[i];
        sum_rolloff += rolloff_vals[i];
        sum_flatness += flatness_vals[i];
        sum_band_ratio += band_ratio_vals[i];
    }
    float mean_zcr = sum_zcr / num_frames;
    float std_zcr = sqrt(sum_zcr_sq / num_frames - mean_zcr * mean_zcr);
    float mean_rms = sum_rms / num_frames;
    float mean_entropy = sum_entropy / num_frames;
    float mean_centroid = sum_centroid / num_frames;
    float mean_rolloff = sum_rolloff / num_frames;
    float mean_flatness = sum_flatness / num_frames;
    float mean_band_ratio = sum_band_ratio / num_frames;

    // Construct the aggregated feature vector.
    // Order: zcr_mean, zcr_std, rms_mean, rms_max, entropy_mean,
    //        spectral_centroid_mean, spectral_rolloff_mean, spectral_flatness_mean, band_ratio_mean.
    feature_vector[0] = mean_zcr;
    feature_vector[1] = std_zcr;
    feature_vector[2] = mean_rms;
    feature_vector[3] = max_rms;
    feature_vector[4] = mean_entropy;
    feature_vector[5] = mean_centroid;
    feature_vector[6] = mean_rolloff;
    feature_vector[7] = mean_flatness;
    feature_vector[8] = mean_band_ratio;

    printf("Feature vector:\n");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f ", feature_vector[i]);
    }
    printf("\n");
    free(zcr_vals); free(rms_vals); free(entropy_vals); free(centroid_vals);
    free(rolloff_vals); free(flatness_vals); free(band_ratio_vals); free(mag);
}

// ==========================
// Hyperdimensional Encoding Functions
// ==========================

// Quantize a normalized value (assumed in [0,1]) into an integer level.
int quantize(float value, int levels) {
    int level = (int)(value * levels);
    if (level >= levels) level = levels - 1;
    if (level < 0) level = 0;
    return level;
}

// Encode the aggregated feature vector into a binary hypervector.
// The process includes encoding each individual feature (via XOR of its codebook and value-level HV)
// and also encoding selected feature pairs, then bundling via majority vote.
void encode_feature_vector(const float *features, bool *hv) {
    // void encode_feature_vector(const float *features, uint8_t *hv) {
    // Temporary storage for individual encoded hypervectors.
    // Total count = NUM_FEATURES + NUM_PAIRS.
    int total_hvs = NUM_FEATURES + NUM_PAIRS;
    // Allocate a 2D array: total_hvs x D (using dynamic allocation).
    int i, j;
    bool **temp_hvs = (bool **)malloc(total_hvs * sizeof(bool *));
    for (i = 0; i < total_hvs; i++) {
        temp_hvs[i] = (bool *)malloc(D * sizeof(bool));
        // Initialize to zero.
        for (j = 0; j < D; j++) {
            temp_hvs[i][j] = 0;
        }
    }

    // Encode each individual feature.
    for (i = 0; i < NUM_FEATURES; i++) {
        int q = quantize(features[i], LEVELS);
        for (j = 0; j < D; j++) {
            // XOR the codebook for feature i with the value-level HV for quantized value.
            temp_hvs[i][j] = codebook[i][j] ^ value_level_hvs[q][j];
        }
    }

    // Encode important feature pairs.
    for (i = 0; i < NUM_PAIRS; i++) {
        int idx = NUM_FEATURES + i;
        int f1 = important_pairs[i].f1;
        int f2 = important_pairs[i].f2;
        int q1 = quantize(features[f1], LEVELS);
        int q2 = quantize(features[f2], LEVELS);
        for (j = 0; j < D; j++) {
            bool hv1 = codebook[f1][j] ^ value_level_hvs[q1][j];
            bool hv2 = codebook[f2][j] ^ value_level_hvs[q2][j];
            temp_hvs[idx][j] = hv1 ^ hv2;
        }
    }

    // Bundle all hypervectors using a majority vote.
    for (j = 0; j < D; j++) {
        int sum = 0;
        for (i = 0; i < total_hvs; i++) {
            sum += temp_hvs[i][j];
        }
        hv[j] = (sum > (total_hvs / 2)) ? 1 : 0;
    }

    // Free temporary storage.
    for (i = 0; i < total_hvs; i++) {
        free(temp_hvs[i]);
    }
    free(temp_hvs);
}

// Compute the Hamming distance between two binary hypervectors.
int hamming_distance(bool *hv1, const bool *hv2, int d) {
    int dist = 0;
    for (int i = 0; i < d; i++) {
        if (hv1[i] != hv2[i]) {
            dist++;
        }
    }
    return dist;
}

// ==========================
// Main Inference Function
// ==========================
int main(void) {
    // -----------------------------
    // 1. Load Audio Data from Storage
    // -----------------------------
    float *audio_data = NULL;
    // copy the audio data from the audio_data.h file
    int audio_length = AUDIO_LENGTH;
    audio_data = (float *)malloc(sizeof(float) * AUDIO_LENGTH);
    for (int i = 0; i < AUDIO_LENGTH; i++) {
        audio_data[i] = audio_data_vector[i];
    }

    printf("Loaded audio data of length: %d\n", audio_length);
    // -----------------------------
    // 2. Extract Aggregated Feature Vector
    // -----------------------------
    float features[NUM_FEATURES] = {0};
    // float features[NUM_FEATURES] = {0.45810546875,
    //     0.049704377129124865,
    //     0.0002689302093737448,
    //     0.0003675652842503041,
    //     2.071158281144997,
    //     1727.1224467060458,
    //     3006.7741297468356,
    //     0.58361894,
    //     3.0681803};
    extract_features(audio_data, audio_length, features);
    free(audio_data);
    printf("Extracted features:\n");
    // -----------------------------
    // 3. Normalize Features (using stored min & range)
    // -----------------------------
    float norm_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        norm_features[i] = (features[i] - feature_min[i]) / feature_range[i];
        if (norm_features[i] < 0.0f) norm_features[i] = 0.0f;
        if (norm_features[i] > 1.0f) norm_features[i] = 1.0f;
    }

    // -----------------------------
    // 4. Encode Feature Vector into a Hypervector
    // -----------------------------
    bool hv[D] = {false};
    encode_feature_vector(norm_features, hv);

    // -----------------------------
    // 5. Perform Inference by Comparing with Stored Class HVs
    // -----------------------------
    int predicted_class = -1;
    int min_distance = D + 1;
    for (int i = 0; i < NUM_CLASSES; i++) {
        int dist = hamming_distance(hv, class_hvs[i], D);
        printf("Class %s: Hamming distance = %d\n", label_names[i], dist);
        if (dist < min_distance) {
            min_distance = dist;
            predicted_class = i;
        }
    }
    printf("Predicted class: %s (Hamming distance: %d)\n", label_names[predicted_class], min_distance);
    // print memory taken by codebook, value_level_hvs and class_hvs
    printf("Memory taken by HVs: %lu bytes\n", sizeof(codebook) + sizeof(value_level_hvs) + sizeof(class_hvs) + sizeof(label_names));

    return 0;
}
