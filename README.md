Ernesto Torres
# Tagalog-English (Taglish) Sentiment & Sarcasm Analyzer

An enhanced sentiment and sarcasm analysis tool for Tagalog and Taglish (Tagalog-English code-switched) text. It runs completely with **zero third-party library dependencies** and uses `Vader.txt` as a standalone lexicon library.

## Overview

**Taglish Sentiment Analyzer** combines a custom text-file lexicon (`Vader.txt`) and built-in VADER calculation rules with Taglish-specific processing:

- **Standalone Lexicon Library (`Vader.txt`)**: Text-file based lexicon with valence ratings from `-4.0` to `+4.0`.
- **Code-switching**: Seamlessly handles Tagalog, Taglish, and English terms.
- **Tagalog Negators**: `hindi`, `di`, `wala`, `ayaw`, `huwag`, `wag`, `dili`, `not`, `no`, `never`.
- **Tagalog Boosters / Intensifiers**: `sobrang`, `napaka`, `grabe`, `medyo`, `konti`.
- **Morphological Prefix Handling**: Decomposes superlative prefixes like `napaka-`.
- **Multi-word Expressions**: Joins idioms like `sana all`, `pwede na yan`, or `walang kwenta` into unified tokens.
- **Rule-Based Sarcasm & Irony Detection**: Via lexical markers, contrast patterns, and mixed-polarity detection.

---

## Features & Architecture

### 1. Standalone Lexicon Library (`Vader.txt`)
- Reads lexicon entries (`<word_or_phrase> <score>`) line-by-line from `Vader.txt` on initialization.
- Supports comments starting with `#` and multi-word phrases (e.g. `walang kwenta -3.0`, `sana all 1.5`, `pwede na yan 0.9`).
- Can be edited directly using any text editor to expand the analyzer's vocabulary.

### 2. Pure-Python VADER Engine (`SentimentIntensityAnalyzer`)
- Built-in polarity scoring with **zero external libraries required** (no `pip install` required).
- Replicates standard VADER rules:
  - **ALL CAPS Emphasis**: Increases valence score by `0.733`.
  - **3-Word Preceding Context**: Checks prior 3 words for boosters (`+0.293` / `-0.293`) and negators (`x -0.74`).
  - **Punctuation Amplification**: Exclamation marks (`!`) boost overall score by `+0.292` (up to 4 `!`).
  - **Compound Score Normalization**:
    $$\text{compound} = \frac{\text{sum\_scores}}{\sqrt{\text{sum\_scores}^2 + 15.0}}$$

### 3. Text Preprocessing (`_prepare_text_for_vader`)
- **Prefix Decomposition**: Splits prefix expressions such as `napakaganda` into `napaka ganda` so `napaka` acts as a booster and `ganda` receives its valence score.
- **Multi-Word Idioms**: Converts phrases like `walang kwenta` into underscored tokens (`walang_kwenta`) so VADER treats them as single lexicon entries.

### 4. Sarcasm & Irony Detection (`SarcasmDetector`)
Detects Taglish sarcasm using three main mechanisms:
1. **Lexical Sarcastic Markers**: Flags expressions like `wow ha`, `edi wow`, `charot`, `eme`, `chos`, `weh di nga` (+0.45 confidence).
2. **Structural Contrast Patterns**: Uses regular expressions to detect positive praise juxtaposed with insults (e.g., `"ganda mo, mukha kang ewan"`) or sarcastic gratitude (e.g., `"salamat ah, nasira mo ang kotse ko"`).
3. **Mixed Polarity Detection**: Flags cases where overall raw sentiment is positive ($\text{compound} > 0.3$) without a concessive conjunction (`pero`, `but`, `however`), but contains an un-negated strong negative term (`tanga`, `bobo`, `sira`, `basura`).

### 5. Score Adjustment Logic
When sarcasm is detected ($\text{confidence} \ge 0.40$), the compound score is adjusted to reflect negative intent:

$$\text{adjusted\_compound} = \begin{cases} -|\text{raw\_compound}| - 0.2 & \text{if } \text{raw\_compound} > 0 \\ -0.35 & \text{if } \text{raw\_compound} = 0 \\ \min(\text{raw\_compound} - 0.25, -0.45) & \text{if } \text{raw\_compound} < 0 \end{cases}$$

Final labels are assigned based on the adjusted compound score:
- **`SARCASTIC / NEGATIVE`**: If sarcasm is detected.
- **`POSITIVE`**: Adjusted compound $\ge +0.05$.
- **`NEGATIVE`**: Adjusted compound $\le -0.05$.
- **`NEUTRAL`**: Between $-0.05$ and $+0.05$.

---

## Installation & Prerequisites

This analyzer runs out-of-the-box on **standard Python 3.7+ with no external pip installations required**.

```bash
python tagalog_english_sentiment.py
```

---

## Quick Start / Code Usage

Import and use `TaglishSentiment` in Python:

```python
from tagalog_english_sentiment import TaglishSentiment

# Initialize analyzer (loads Vader.txt by default)
analyzer = TaglishSentiment(lexicon_file="Vader.txt")

# Analyze a sample text
result = analyzer.analyze("Sobrang galing ng palabas!")
print(result)
# Output:
# {
#     'text': 'Sobrang galing ng palabas!',
#     'raw_scores': {'neg': 0.0, 'neu': 0.51, 'pos': 0.49, 'compound': 0.7783},
#     'adjusted_compound': 0.7783,
#     'is_sarcastic': False,
#     'sarcasm_confidence': 0.0,
#     'sarcasm_reasons': [],
#     'label': 'POSITIVE'
# }

# Sarcastic text example
sarcastic_res = analyzer.analyze("Wow ha, ikaw na ang magaling.")
print(sarcastic_res['label'], sarcastic_res['adjusted_compound'])
# Output: SARCASTIC / NEGATIVE -0.7859
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

- [Vader.txt](file:///c:/Users/PC/Documents/VaderSentiLexi/Vader.txt): The text-file lexicon library containing words, phrases, and sentiment scores.
- [tagalog_english_sentiment.py](file:///c:/Users/PC/Documents/VaderSentiLexi/tagalog_english_sentiment.py): Standalone pure-Python VADER engine, `SarcasmDetector`, `TaglishSentiment` analyzer, and CLI loop.
- [README.md](file:///c:/Users/PC/Documents/VaderSentiLexi/README.md): Project documentation and usage guide.
