# Tagalog-English (Taglish) Sentiment & Sarcasm Analyzer

An enhanced sentiment and sarcasm analysis tool for Tagalog and Taglish (Tagalog-English code-switched) text, built on top of [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment).

---

## Overview

Standard VADER is designed for English texts. **Taglish Sentiment Analyzer** extends VADER to handle the unique linguistic patterns of Taglish, including:

- **Code-switching** (mixing Tagalog and English words seamlessly).
- **Custom Tagalog Sentiment Lexicon** with valence ratings from `-4.0` to `+4.0`.
- **Tagalog Negators** (`hindi`, `di`, `wala`, `ayaw`, `huwag`, `wag`).
- **Tagalog Boosters / Intensifiers** (`sobrang`, `napaka`, `grabe`, `medyo`, `konti`).
- **Morphological Prefix Handling** (decomposing superlative prefixes like `napaka-`).
- **Multi-word Expressions** (joining idioms like `sana all` or `walang kwenta` into unified tokens).
- **Rule-Based Sarcasm & Irony Detection** via lexical markers, contrast patterns, and mixed-polarity detection.

---

## Features & Architecture

### 1. Taglish Lexicon & Modifiers
- Injects a dictionary of common Tagalog positive and negative terms into VADER's polarity engine.
- Registers Tagalog negators into VADER's `NEGATE` set to properly invert scores (e.g., `"hindi maganda"` is scored negative).
- Adds Tagalog intensifiers (`B_INCR`) and de-amplifiers (`B_DECR`) to VADER's `BOOSTER_DICT`.

### 2. Text Preprocessing (`_prepare_text_for_vader`)
- **Prefix Decomposition**: Splits prefix expressions such as `napakaganda` into `napaka ganda` so `napaka` acts as a booster and `ganda` receives its valence score.
- **Multi-Word Idioms**: Converts phrases like `walang kwenta` into underscored tokens (`walang_kwenta`) so VADER treats them as single lexicon entries.

### 3. Sarcasm & Irony Detection (`SarcasmDetector`)
Detects Taglish sarcasm using three main mechanisms:
1. **Lexical Sarcastic Markers**: Flags expressions like `wow ha`, `edi wow`, `charot`, `eme`, `chos`, `weh di nga` (+0.45 confidence).
2. **Structural Contrast Patterns**: Uses regular expressions to detect positive praise juxtaposed with insults (e.g., `"ganda mo, mukha kang ewan"`) or sarcastic gratitude (e.g., `"salamat ah, nasira mo ang kotse ko"`).
3. **Mixed Polarity Detection**: Flags cases where the overall raw sentiment is positive ($\text{compound} > 0.3$) without a concessive conjunction (`pero`, `but`, `however`), but contains an un-negated strong negative term (`tanga`, `bobo`, `sira`, `basura`).

### 4. Score Adjustment Logic
When sarcasm is detected ($\text{confidence} \ge 0.40$), the compound score is adjusted to reflect the negative intent:

$$\text{adjusted\_compound} = \begin{cases} -|\text{raw\_compound}| - 0.2 & \text{if } \text{raw\_compound} > 0 \\ -0.35 & \text{if } \text{raw\_compound} = 0 \\ \min(\text{raw\_compound} - 0.25, -0.45) & \text{if } \text{raw\_compound} < 0 \end{cases}$$

Final labels are assigned based on the adjusted compound score:
- **`SARCASTIC / NEGATIVE`**: If sarcasm is detected.
- **`POSITIVE`**: Adjusted compound $\ge +0.05$.
- **`NEGATIVE`**: Adjusted compound $\le -0.05$.
- **`NEUTRAL`**: Between $-0.05$ and $+0.05$.

---

## Installation & Prerequisites

Ensure you have Python 3.7+ installed. Install the `vaderSentiment` dependency:

```bash
pip install vaderSentiment
```

---

## Quick Start / Code Usage

You can import and use `TaglishSentiment` in Python:

```python
from tagalog_english_sentiment import TaglishSentiment

# Initialize analyzer
analyzer = TaglishSentiment()

# Analyze a sample text
result = analyzer.analyze("Sobrang galing ng palabas!")
print(result)
# Output:
# {
#     'text': 'Sobrang galing ng palabas!',
#     'raw_scores': {'neg': 0.0, 'neu': 0.448, 'pos': 0.552, 'compound': 0.6792},
#     'adjusted_compound': 0.6792,
#     'is_sarcastic': False,
#     'sarcasm_confidence': 0.0,
#     'sarcasm_reasons': [],
#     'label': 'POSITIVE'
# }

# Sarcastic text example
sarcastic_res = analyzer.analyze("Wow ha, ikaw na ang magaling.")
print(sarcastic_res['label'], sarcastic_res['adjusted_compound'])
# Output: SARCASTIC / NEGATIVE -0.8792
```

---

## Interactive CLI Commands

Run the script directly to launch the interactive command line tool:

```bash
python tagalog_english_sentiment.py
```

### CLI Commands Reference

| Command | Description | Example |
| :--- | :--- | :--- |
| `add <word> <score>` | Add or update a lexicon word/phrase with a score from `-4.0` to `+4.0`. | `add lodi 2.5` |
| `negator <word>` | Register a custom negation word. | `negator hindi` |
| `booster <word> [incr\|decr]` | Add a booster/intensifier or de-amplifier word. | `booster sobra incr` |
| `sarcasm <phrase>` | Add a new sarcastic trigger phrase. | `sarcasm edi shing` |
| `remove [type] <word/phrase>` | Remove an entry from `lexicon`, `negator`, `booster`, `sarcasm`, or automatically search all types. | `remove negator dili` / `remove maganda` |
| `quit` / `exit` / `q` | Exit the CLI application. | `quit` |

---

## File Structure

- `tagalog_english_sentiment.py`: Primary source code containing `SarcasmDetector`, `TaglishSentiment`, dynamic lexicon helpers, and the CLI `main()` loop.
- `README.md`: Project documentation and usage guide.
