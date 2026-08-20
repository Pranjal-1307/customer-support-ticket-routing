# Preprocessing Audit Report (Milestone 1)

**Generated:** 2026-08-20T06:12:50.188271

## Pipeline Steps (utils/preprocess.py)

1. **Input validation** — Returns empty string for non-string or whitespace-only input
2. **Lowercase** — `text.lower()`
3. **URL removal** — Regex: `https?://\S+|www\.\S+`
4. **HTML tag removal** — Regex: `<.*?>`
5. **Special character + digit removal** — Regex: `[^a-z\s]` → only lowercase letters and spaces remain
6. **Whitespace normalization** — Collapse multiple spaces, strip
7. **Tokenization** — NLTK `word_tokenize` (with fallback to `.split()`)
8. **Stopword removal** — NLTK English stopwords, also filters tokens with len ≤ 1
9. **Lemmatization** — WordNet lemmatizer (default POS=noun)
10. **Join** — Space-separated string output

## Verification Test Cases

| Input | Output | Status |
|-------|--------|--------|
| `''` | `` | ✅ Empty input handled |
| `'   '` | `` | ✅ Whitespace-only handled |
| `None` | `` | ✅ None input handled |
| `123` | `` | ✅ Non-string handled |
| `'My CREDIT card was CHARGED twice!'` | `credit card charged twice` | ✅ Lowercase + punctuation |
| `'Check https://example.com/help for info'` | `check info` | ✅ URL removed |
| `'Error <b>500</b> occurred'` | `error occurred` | ✅ HTML + digits removed |
| `'  multiple   spaces   here  '` | `multiple space` | ✅ Whitespace normalized |
| `'I am running and jumping'` | `running jumping` | ✅ Stopwords removed |

## Training ↔ Inference Consistency

- **Training** (`train_models.py` line 42): `train_df['processed_text'] = train_df['ticket_text'].apply(preprocess_text)`
- **Inference** (`predict.py` line 77): `processed = preprocess_text(ticket_text)`
- **Same function used:** ✅ Yes — both import from `utils.preprocess.preprocess_text`
- **No train-time-only transforms:** ✅ Correct

## Observations

- The pipeline is appropriate for TF-IDF bag-of-words classification.
- Lemmatization uses default noun POS tag, which may not lemmatize verbs optimally (e.g., 'running' may stay 'running'). This is acceptable for M1.
- All digits are removed, which means order IDs, error codes, and amounts are stripped. This is acceptable since these are random fills in synthetic data.
- No stemming is applied (lemmatization only). This is a design choice, not an error.

**Status: ✅ CORRECT — No changes needed.**