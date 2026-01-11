# Prediction-Reference Baseline (LSTM Predictor)

## Overview
This baseline is a **sequence-to-sequence LSTM predictor** that learns to predict the **post window** from the **pre window** within each **mother segment**. After training on clean data, the same predictor is used to generate predicted routes for a clean reference (base) and for target routes (val/test). Anomalies are flagged by **disagreement between predicted routes** rather than by direct reconstruction error. Train thresholds and summary stats now use a **post-only prediction error** on the training windows to provide a direct view of prediction quality.

Key idea:
- **Mother segment windowing**: each route is split into three macro segments, then into sliding windows (mother segments). Each window is split into **pre** and **post** halves.
- **Prediction task**: the LSTM is trained to predict **post** from **pre**.
- **Scoring**: base vs target predicted routes are compared window-by-window. Large disagreement implies anomalous behavior.

## Data Preparation and Mother Segment Windowing
1. **Macro segmentation**: A route is divided into three macro segments based on its length.
2. **Mother segments**: Within each macro segment, sliding windows (length = `L`, stride = `S`) are extracted.
3. **Pre/Post split**: Each window is split at the midpoint into:
   - `pre`: first half of the window
   - `post`: second half of the window

This is the fundamental unit for training and scoring. The model does not directly operate on full routes; it operates on these **mother segments**.

## Model: LSTM Predictor
The model is a simple **sequence-to-sequence LSTM** with a linear head:
- Input: `pre` sequence (shape: `[window_len/2, feature_dim]`)
- Output: predicted `post` sequence (same shape)

This model is trained only on clean data, so it learns **normal predictive patterns**.

## Training Process (Learn Predictive Ability)
**Goal:** learn to predict `post` from `pre` accurately.

- **Loss function**: Mean Squared Error (MSE)

  \[
  \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{Y}_i - Y_i \rVert^2
  \]

  where `Y_i` is the true `post` window and `\hat{Y}_i` is the predicted `post`.

- **Behavioral meaning**:
  - Minimizing MSE teaches the LSTM to capture normal temporal dependencies between `pre` and `post`.
  - This is **not** an anomaly score by itself; it is purely for learning the predictive dynamics.

## Validation Process (Select Threshold)
**Goal:** choose a decision threshold for anomaly scoring.

1. Generate predicted routes for:
   - **Base route** (clean training data only)
   - **Validation route**
2. **Align by macro segment** so predictions are matched within the same macro section.
3. Compute **window-level disagreement scores** (see below).
4. Use **percentile-based thresholding** on validation scores:

  \[
  T = \text{percentile}(S_{\text{val}}, p)
  \]

  where `p` is a chosen percentile (e.g., 99.5). This yields a threshold `T`.

- **Behavioral meaning**:
  - Validation does not retrain the model.
  - It simply determines how large a disagreement score must be before we call it anomalous.

## Test Process (Compute Scores)
**Goal:** compute anomaly scores and evaluate.

1. Generate predicted routes for:
   - **Base route** (clean training reference)
   - **Test route**
2. Align by macro segment.
3. Compute window-level disagreement scores.
4. Map window scores back to **timepoint scores**.
5. Apply the threshold from validation to produce binary anomaly predictions.

## Scoring Details
### 1) Window-level disagreement (mean vector L2)
For each aligned window pair (`base_pred`, `test_pred`):

1. Compute mean vectors across time for each window:
   \[
   \bar{Y}^{\text{base}} = \frac{1}{T}\sum_t \hat{Y}^{\text{base}}_t,\quad
   \bar{Y}^{\text{test}} = \frac{1}{T}\sum_t \hat{Y}^{\text{test}}_t
   \]
2. Compute the L2 norm of the mean-vector difference:
   \[
   s_{\text{window}} = \lVert \bar{Y}^{\text{base}} - \bar{Y}^{\text{test}} \rVert_2
   \]

**Meaning:** a window is anomalous if the average predicted post behavior deviates between base and test, which is more robust to single-step spikes.

### 1b) Train window prediction error (post-only mean-vector L2)
For train windows, we compute a direct prediction error against the **ground-truth post** window:

1. Compute mean vectors across time for prediction and ground truth:
   \[
   \bar{Y}^{\text{pred}} = \frac{1}{T}\sum_t \hat{Y}^{\text{train}}_t,\quad
   \bar{Y}^{\text{true}} = \frac{1}{T}\sum_t Y^{\text{train}}_t
   \]
2. Compute the L2 norm of the mean-vector difference:
   \[
   s^{\text{train}}_{\text{window}} = \lVert \bar{Y}^{\text{pred}} - \bar{Y}^{\text{true}} \rVert_2
   \]

**Meaning:** this measures **post-only prediction error** for training windows using averaged post vectors and is used to derive train-based thresholds and stats.

### 2) Mapping window scores to timepoint scores
Each mother segment score is mapped back onto its original time indices. If multiple windows cover the same timepoint, the **maximum** score is kept:

\[
S_{\text{time}}[t] = \max_{\text{windows covering } t} s_{\text{window}}
\]

**Meaning:** every timepoint inherits the strongest anomaly evidence from any window that contains it.

## Summary of Processes and Their Scoring Meaning
- **Training**: learns predictive capability using MSE loss; **not** anomaly scoring.
- **Training stats**: computes post-only prediction error on train windows to summarize train quality and derive train-based thresholds.
- **Validation**: computes window disagreement scores and sets threshold by percentile.
- **Testing**: computes window disagreement scores, maps them to timepoints, and applies threshold for anomaly detection (including optional evaluation with the train-derived threshold).

## Final Evaluation Against Human Labels (Four Metrics)
After obtaining **timepoint anomaly predictions** (by applying the threshold), the system compares them with **human-labeled anomaly ground truth** and computes four standard metrics:

1. **Precision**  
   - **Meaning**: among all points predicted as anomalous, how many are truly anomalous.  
   - High precision means **few false positives**.

2. **Recall**  
   - **Meaning**: among all truly anomalous points, how many are correctly detected.  
   - High recall means **few false negatives**.

3. **F1-score**  
   - **Meaning**: harmonic mean of precision and recall, balancing both.  
   - Useful when you want a **single score** that trades off false positives and false negatives.

4. **PR-AUC (Average Precision)**  
   - **Meaning**: area under the Precision–Recall curve, summarizing performance across all thresholds.  
   - Robust when anomalies are rare, reflecting **overall ranking quality** of anomaly scores.

## Key Takeaways
- The LSTM is trained on clean data to predict post-windows.
- Anomaly scores are **not reconstruction errors**, but **disagreement between predicted routes**.
- Scoring is window-based (mean-vector L2), then mapped to timepoints by max aggregation.
- Validation chooses a threshold; testing applies it to produce anomalies.
- The smoothing step reports how many windows were selected and concatenated to form the training subset.
