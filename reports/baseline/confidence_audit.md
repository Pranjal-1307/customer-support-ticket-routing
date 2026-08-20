# Confidence Score Audit Report (Milestone 1)

**Generated:** 2026-08-20T06:13:18.502345

## How Confidence is Calculated (`utils/predict.py`, lines 84-98)

The code follows a cascade:

### 1. Models with `predict_proba` (Logistic Regression, Naive Bayes, Random Forest)

```python
probs = self.ticket_classifier.predict_proba(vec_input)[0]
pred_idx = np.argmax(probs)
confidence = float(probs[pred_idx])
```
- Returns the maximum class probability from the model's probability distribution.
- **Logistic Regression:** Probabilities from logistic function — reasonably calibrated by default.
- **Naive Bayes:** Probabilities can be overconfident due to independence assumptions.
- **Random Forest:** Probabilities from vote fractions — tends to be well-calibrated.

### 2. Models with `decision_function` (LinearSVC)

```python
dec = self.ticket_classifier.decision_function(vec_input)[0]
exp_dec = np.exp(dec - np.max(dec))
probs = exp_dec / exp_dec.sum()
pred_idx = np.argmax(probs)
confidence = float(probs[pred_idx])
```
- Applies **softmax** to raw decision function margins.
- ⚠️ **This is NOT a calibrated probability.** The softmax normalizes margins to [0,1] sum-to-1, but the resulting values do not represent true class probabilities.
- The margins are unbounded and their magnitude depends on regularization, feature scale, and class separability.
- A softmax confidence of 95% from LinearSVC is **not comparable** to a 95% probability from Logistic Regression.

### 3. Fallback (no `predict_proba` or `decision_function`)

```python
confidence = 1.0
```
- Returns 100% confidence — clearly incorrect but only used as a last resort.

## Current Production Model

The best model selected is typically **Logistic Regression**, which has native `predict_proba`.
Therefore the current production confidence is a Logistic Regression probability, which is **acceptable** though not formally calibrated.

## Priority Confidence

The priority classifier is **Logistic Regression**, which uses `predict_proba`.
However, the prediction code (`predict.py` line 103) only calls `predict()` — **priority confidence is never calculated or returned**.
The returned `confidence` field represents **category confidence only**.

## Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| LogReg confidence | ✅ Acceptable | Native probability, not formally calibrated |
| NB confidence | ⚠️ Caution | Can be overconfident |
| LinearSVC confidence | ⚠️ Misleading | Softmax of margins ≠ probability |
| RF confidence | ✅ Acceptable | Vote fraction |
| Priority confidence | ❌ Missing | Not computed or returned |
| Confidence label | ⚠️ Imprecise | Presented as percentage but not formally calibrated |

## Recommendations (for later milestones)

1. Add `CalibratedClassifierCV` wrapper for LinearSVC if it becomes the production model.
2. Compute and return separate priority confidence.
3. Label confidence as 'model score' rather than implying calibrated probability.
4. Apply Platt scaling or isotonic calibration for production confidence.

**M1 Status: DOCUMENTED — No code changes needed.** The current production model (LogReg) provides reasonable probability estimates. The LinearSVC softmax issue is documented.