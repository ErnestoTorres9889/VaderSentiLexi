# Tagalog-English (Taglish) Sentiment Analyzer

An extended **VADER-based Sentiment Intensity Analyzer** tailored for Tagalog-English (Taglish) code-switched text. It features custom sentiment lexicon management, dynamic negator and booster registration, and a structural sarcasm detection engine.

---

## 📌 Features

1. **Extended Tagalog Lexicon**:
   - Built-in Tagalog words mapped to sentiment intensity scores ($-4.0$ to $+4.0$).
   - Dynamic methods to add or adjust sentiment scores at runtime.
2. **Negators & Boosters Engine**:
   - Native support for Tagalog negators (`hindi`, `di`, `ayaw`, `wala`, `wag`, etc.) and boosters (`sobrang`, `napaka`, `grabe`, `medyo`, etc.).
   - Dynamically register custom negators or boosters.
3. **Structural Sarcasm Phrase Detection**:
   - **Marker Detection**: Catch common Taglish sarcastic triggers (`wow ha`, `edi ikaw na`, `charot`, `eme`, `pabida`).
   - **Regex Structural Contrast**: Identifies positive praise coupled with insults (*"ganda mo, mukha kang ewan"*) or ironic thanks (*"salamat ah, nasira mo..."*).
   - **Mixed Polarity Contrast**: Detects sentences with positive compound scores containing negative context words.
   - **Score Inversion**: Automatically adjusts compound scores and labels sarcastic sentences as `SARCASTIC / NEGATIVE`.

---

## 🚀 Installation & Requirements

### Prerequisites
- Python 3.8+
- `vaderSentiment` package

### Installation
```bash
pip install vaderSentiment
```

---

## 🏗️ Code Architecture & Detailed Breakdown

The codebase is split into two primary components: `SarcasmDetector` and `TaglishSentiment`.

```
                  ┌──────────────────────────────┐
                  │    User Input (Taglish)      │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Text Normalization        │
                  └──────────────┬───────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌─────────────────────────────┐ ┌─────────────────────────────┐
  │  VADER Engine + Tagalog     │ │     SarcasmDetector         │
  │  Lexicon/Negators/Boosters  │ │  (Markers, Regex Patterns,  │
  │                             │ │   Mixed Polarity Contrast)  │
  └──────────────┬──────────────┘ └──────────────┬──────────────┘
                 │                               │
                 │ Raw Scores                    │ Sarcasm Detected?
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Sentiment Score Adjustment   │
                  │ & Labeling Pipeline          │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Result (Label & Compound)    │
                  └──────────────────────────────┘
```

---

### 1. `SarcasmDetector` Class

Analyzes text for structural sarcasm, ironic markers, and polarity contradictions.

#### **Key Components**:
* **`self.sarcastic_phrases`**: List of explicit Taglish sarcasm triggers (`wow ha`, `edi ikaw na`, `pabida`, `charot`, `eme`, `wehh`).
* **`self.contrast_patterns`**: Regular expressions detecting structural irony:
  - `(ganda|galing|bait)\b.*(mukha|ewan|tanga|bobo|sira)`: Positive praise followed by an insult.
  - `salamat\s+(ah|ha)?.*(nasira|tinapon|pinagpalit)`: Sarcastic gratitude followed by damage.
  - `(edi|sige)\s+ikaw\s+na\b`: Mocking praise phrase.
  - `["\'](galing|mabuti|maganda)["\']`: Quotes around praise words to indicate irony.

#### **Detection Method (`detect(text, initial_compound)`)**:
Returns `(is_sarcastic: bool, confidence: float, reasons: List[str])`:
1. Increments score by `+0.45` when explicit sarcastic markers are found.
2. Increments score by `+0.50` when structural regex patterns match.
3. Increments score by `+0.40` if initial sentiment is positive (`compound > 0.3`) but negative context words exist (`ewan`, `sira`, `bobo`).
4. Increments score by `+0.35` for interjection patterns (`wow ... ha` or `galing ... ah`).
5. Marks `is_sarcastic = True` if total score $\ge 0.40$.

---

### 2. `TaglishSentiment` Class

Wraps VADER's `SentimentIntensityAnalyzer` and extends its internal lookup dictionaries.

#### **Key Dictionaries**:
- **`DEFAULT_LEXICON`**: Dictionary mapping Tagalog words to intensity scores ($-4.0$ to $+4.0$).
  - *Positive*: `"maganda": 2.7`, `"magaling": 2.8`, `"masaya": 2.9`, `"astig": 2.4`, `"lodi": 2.5`
  - *Negative*: `"pangit": -2.7`, `"bobo": -3.0`, `"tanga": -3.2`, `"basura": -3.0`, `"bwisit": -2.8`
- **`DEFAULT_NEGATORS`**: List of negation words (`hindi`, `di`, `ayaw`, `wala`, `huwag`, `wag`).
- **`DEFAULT_BOOSTERS`**: Dictionary of intensifier words (`sobrang`, `napaka`, `grabe`, `medyo`).

#### **Extension Method (`_initialize_vader_extension()`)**:
Appends Tagalog data into VADER's native tables:
```python
self.analyzer.lexicon.update(self.DEFAULT_LEXICON)
for word in self.DEFAULT_NEGATORS:
    if word not in vs.NEGATE:
        vs.NEGATE.append(word)
vs.BOOSTER_DICT.update(self.DEFAULT_BOOSTERS)
```

#### **Dynamic Management API**:
- **`add_lexicon_word(word: str, score: float)`**: Adds or modifies a word score in the lexicon.
- **`add_negator(word: str)`**: Appends a new negator word to `vs.NEGATE`.
- **`add_booster(word: str, modifier: float)`**: Adds a new intensifier to `vs.BOOSTER_DICT`.
- **`add_sarcastic_phrase(phrase: str)`**: Registers a custom sarcasm trigger in `SarcasmDetector`.

#### **Sentiment Analysis & Score Adjustment (`analyze(text)`)**:
1. Normalizes input text (`_clean`).
2. Calculates raw VADER scores (`pos`, `neu`, `neg`, `compound`).
3. Runs sarcasm detection.
4. **Score Inversion Logic**:
   If sarcasm is detected and raw `compound > 0`, the compound score is inverted:
   $$\text{adjusted\_compound} = -|\text{raw\_compound}| - 0.2$$
5. Determines final label (`POSITIVE`, `NEGATIVE`, `NEUTRAL`, or `SARCASTIC / NEGATIVE`).

---

## 💻 Python Usage Example

```python
from tagalog_english_sentiment import TaglishSentiment

# Initialize analyzer
analyzer = TaglishSentiment()

# 1. Standard Sentiment Analysis
result = analyzer.analyze("Maganda at mabuti ang araw na ito.")
print(result["label"])  # POSITIVE
print(result["adjusted_compound"])  # 0.7717

# 2. Sarcasm Detection
sarc_result = analyzer.analyze("Wow ha, ikaw na ang magaling.")
print(sarc_result["label"])  # SARCASTIC / NEGATIVE
print(sarc_result["is_sarcastic"])  # True
print(sarc_result["sarcasm_reasons"])  # ["Detected sarcastic marker: 'wow ha'", ...]

# 3. Dynamic Lexicon & Booster Management
analyzer.add_lexicon_word("petmalu", 3.0)
analyzer.add_booster("tindi", 0.293)
analyzer.add_negator("dili")
```

---

## ⌨️ Interactive CLI Commands

Run `python tagalog_english_sentiment.py` to enter interactive mode:

| Command | Syntax | Description |
| :--- | :--- | :--- |
| **Add Lexicon Word** | `add <word> <score>` | Adds a word with score between `-4.0` and `+4.0` |
| **Add Negator** | `negator <word>` | Registers a word as a sentiment negator |
| **Add Booster** | `booster <word> [incr\|decr]` | Registers an intensifier (`incr` or `decr`) |
| **Add Sarcasm Trigger** | `sarcasm <phrase>` | Registers a custom sarcastic phrase trigger |
| **Quit** | `quit` | Exits the CLI program |

---

## 📊 Sample Test Results

| Input Sentence | Label | Compound Score | Notes |
| :--- | :--- | :--- | :--- |
| `"Maganda at mabuti ang araw na ito."` | `POSITIVE` | `+0.7717` | Standard positive sentiment |
| `"Hindi maganda ang lasa ng pagkain."` | `NEGATIVE` | `-0.4585` | Negator `hindi` flips `maganda` |
| `"Sobrang galing ng palabas!"` | `POSITIVE` | `+0.6581` | Booster `sobrang` amplifies `galing` |
| `"Wow ha, ikaw na ang magaling."` | `SARCASTIC / NEGATIVE` | `-1.0000` | Caught markers `wow ha`, `ikaw na` |
| `"Ang ganda mo, mukha kang ewan."` | `SARCASTIC / NEGATIVE` | `-0.5400` | Structural regex praise+insult match |
| `"Salamat ah, nasira mo ang kotse ko."` | `SARCASTIC / NEGATIVE` | `-0.6588` | Sarcastic gratitude pattern match |

