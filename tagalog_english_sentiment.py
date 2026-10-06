"""Tagalog-English (Taglish) sentiment analyzer built on VADER.

Extended with:
1. Lexicon word management (with sentiment scores -4.0 to +4.0).
2. Custom Negator and Booster/Intensifier management.
3. Structural Sarcasm Phrase Analysis for Taglish.

Usage:
    python tagalog_english_sentiment.py
"""

import re
from typing import Dict, List, Tuple, Any
import vaderSentiment.vaderSentiment as vs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SarcasmDetector:
    """Analyzes Taglish text for structural sarcasm patterns and sarcastic markers."""

    def __init__(self):
        # Sarcastic markers and phrases common in Taglish
        self.sarcastic_phrases = [
            "wow ha", "wow ah", "edi ikaw na", "ikaw na", "ikaw na magaling",
            "galing mo talaga", "ganda mo", "sige pa", "pabida", "charot",
            "charr", "eme", "wehh", "salamat ah", "salamat ha", "pagpatuloy mo yan"
        ]

        # Regular expressions for structural sarcasm detection
        self.contrast_patterns = [
            # Positive praise followed by contrast/insult: e.g. "ganda mo, mukha kang ewan"
            r"(ganda|galing|bait|talino|sarap)\b.*(mukha|ewan|tanga|bobo|sira|mali|walang kwenta)",
            # "Salamat" followed by damage/negative result: e.g. "salamat ah, nasira mo"
            r"salamat\s+(ah|ha|din)?.*(nasira|tinapon|pinagpalit|mali|bwisit|tapon)",
            # "Edi/Ikaw na" praise patterns
            r"(edi|sige)\s+ikaw\s+na\b",
            # Quotes around positive praise: e.g. "magaling" ka talaga
            r'["\'](galing|mabuti|maganda|bait|lodi)["\']'
        ]

    def add_sarcastic_phrase(self, phrase: str):
        """Add a custom sarcastic phrase trigger."""
        phrase_clean = phrase.strip().lower()
        if phrase_clean and phrase_clean not in self.sarcastic_phrases:
            self.sarcastic_phrases.append(phrase_clean)

    def detect(self, text: str, initial_compound: float) -> Tuple[bool, float, List[str]]:
        """Detect sarcasm based on structural rules, markers, and sentiment polarity contrast.

        Returns:
            Tuple of (is_sarcastic: bool, confidence_score: float, reasons: List[str])
        """
        text_lower = text.lower().strip()
        reasons = []
        score = 0.0

        # Check explicit sarcastic markers
        for phrase in self.sarcastic_phrases:
            if phrase in text_lower:
                score += 0.45
                reasons.append(f"Detected sarcastic marker: '{phrase}'")

        # Check structural regex contrast patterns
        for pattern in self.contrast_patterns:
            if re.search(pattern, text_lower):
                score += 0.50
                reasons.append(f"Matched sarcastic structural contrast pattern: '{pattern}'")

        # Check structural contrast: positive sentiment words combined with negative context
        if initial_compound > 0.3:
            negative_words = ["ewan", "tanga", "bobo", "sira", "basura", "pangit", "kainis", "nasira", "mali"]
            for neg_word in negative_words:
                if neg_word in text_lower:
                    score += 0.40
                    reasons.append(f"Mixed polarity contrast: positive overall score with negative word '{neg_word}'")
                    break

        # Check sarcastic punctuation structure (e.g., quotes around praise or excessive exclamation after sarcasm phrase)
        if re.search(r'\b(wow|galing|ganda)\b.*\b(ha|ah)\b', text_lower):
            score += 0.35
            reasons.append("Detected sarcastic interjection structure (e.g. 'wow ha' / 'galing ah')")

        is_sarcastic = score >= 0.40
        confidence = min(score, 1.0)
        return is_sarcastic, round(confidence, 2), reasons


class TaglishSentiment:
    """Tagalog-English (Taglish) Sentiment Analyzer extending VADER with custom lexicons,
    negators, boosters, and sarcasm detection.
    """

    # Base Tagalog lexicon mapping words to sentiment scores (-4.0 to +4.0)
    DEFAULT_LEXICON = {
        # Positive words
        "maganda": 2.7, "mabuti": 2.0, "magaling": 2.8, "masaya": 2.9,
        "mahal": 2.5, "salamat": 2.0, "galing": 2.8, "astig": 2.4,
        "masarap": 2.6, "gwapo": 2.2, "ganda": 2.6, "tama": 1.5,
        "nakakatuwa": 2.5, "ayos": 1.8, "swerte": 2.0, "paborito": 2.4,
        "sigurado": 2.2, "panalo": 2.7, "wagi": 2.6, "maaasahan": 2.3,
        "lodi": 2.5, "petmalu": 2.6, "sana all": 1.5, "respeto": 2.0,

        # Negative words
        "pangit": -2.7, "masama": -2.5, "malungkot": -2.6, "galit": -2.8,
        "mahirap": -1.5, "sayang": -1.8, "walang kwenta": -3.0, "basura": -3.0,
        "nakakainis": -2.7, "nakakasuya": -2.2, "bobo": -3.0, "pagod": -1.4,
        "takot": -2.0, "sira": -2.2, "mali": -1.6, "kainis": -2.6,
        "tanga": -3.2, "inutil": -3.1, "kupal": -3.5, "epal": -2.5,
        "pabida": -2.4, "ewan": -1.2, "bwisit": -2.8, "buwisit": -2.8,
    }

    # Tagalog negators
    DEFAULT_NEGATORS = ["hindi", "di", "ayaw", "wala", "huwag", "wag", "hinding-hindi", "dili"]

    # Tagalog boosters / intensifiers
    DEFAULT_BOOSTERS = {
        "sobrang": vs.B_INCR,
        "napaka": vs.B_INCR,
        "grabe": vs.B_INCR,
        "talaga": vs.B_INCR,
        "ubod ng": vs.B_INCR,
        "ubod": vs.B_INCR,
        "masyadong": vs.B_INCR,
        "lalo": vs.B_INCR,
        "medyo": vs.B_DECR,
        "konti": vs.B_DECR,
        "kaunti": vs.B_DECR,
        "bahagya": vs.B_DECR,
    }

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sarcasm_detector = SarcasmDetector()
        self._initialize_vader_extension()

    def _initialize_vader_extension(self):
        """Extend VADER analyzer with initial Tagalog lexicons, negators, and boosters."""
        self.analyzer.lexicon.update(self.DEFAULT_LEXICON)

        for word in self.DEFAULT_NEGATORS:
            if word not in vs.NEGATE:
                vs.NEGATE.append(word)

        vs.BOOSTER_DICT.update(self.DEFAULT_BOOSTERS)

    def add_lexicon_word(self, word: str, score: float) -> float:
        """Add or update a lexicon word with a sentiment score (-4.0 to 4.0)."""
        clean_word = word.strip().lower()
        score = max(-4.0, min(4.0, float(score)))
        self.analyzer.lexicon[clean_word] = score
        return score

    def remove_lexicon_word(self, word: str) -> bool:
        """Remove a word from the active lexicon."""
        clean_word = word.strip().lower()
        if clean_word in self.analyzer.lexicon:
            del self.analyzer.lexicon[clean_word]
            return True
        return False

    def add_negator(self, word: str) -> str:
        """Add a custom negator word."""
        clean_word = word.strip().lower()
        if clean_word not in vs.NEGATE:
            vs.NEGATE.append(clean_word)
        return clean_word

    def add_booster(self, word: str, modifier: float = vs.B_INCR) -> Tuple[str, float]:
        """Add a custom booster / intensifier word with modifier value."""
        clean_word = word.strip().lower()
        vs.BOOSTER_DICT[clean_word] = modifier
        return clean_word, modifier

    def add_sarcastic_phrase(self, phrase: str):
        """Add a custom sarcastic phrase trigger."""
        self.sarcasm_detector.add_sarcastic_phrase(phrase)

    @staticmethod
    def _clean(text: str) -> str:
        """Clean and normalize input text."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment and structural sarcasm.

        Returns detailed dictionary containing:
        - raw sentiment scores
        - sarcasm detection results
        - adjusted compound score
        - final sentiment label
        """
        cleaned_text = self._clean(text)
        scores = self.analyzer.polarity_scores(cleaned_text)
        raw_compound = scores["compound"]

        # Run Sarcasm Detection
        is_sarcastic, sarcasm_conf, reasons = self.sarcasm_detector.detect(cleaned_text, raw_compound)

        # Adjust compound score if sarcasm is detected
        adjusted_compound = raw_compound
        if is_sarcastic:
            if raw_compound > 0:
                # Invert positive sentiment to negative for sarcastic praise
                adjusted_compound = -abs(raw_compound) - 0.2
            elif raw_compound == 0:
                adjusted_compound = -0.35
            else:
                adjusted_compound = raw_compound - 0.15

            # Clamp compound between -1.0 and 1.0
            adjusted_compound = max(-1.0, min(1.0, adjusted_compound))

        label = self._determine_label(adjusted_compound, is_sarcastic)

        return {
            "text": cleaned_text,
            "raw_scores": scores,
            "adjusted_compound": round(adjusted_compound, 4),
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence": sarcasm_conf,
            "sarcasm_reasons": reasons,
            "label": label
        }

    @staticmethod
    def _determine_label(compound: float, is_sarcastic: bool) -> str:
        """Determine final text sentiment label."""
        if is_sarcastic:
            return "SARCASTIC / NEGATIVE"
        if compound >= 0.05:
            return "POSITIVE"
        if compound <= -0.05:
            return "NEGATIVE"
        return "NEUTRAL"


def main():
    tool = TaglishSentiment()
    print("==================================================")
    print("   Tagalog-English (Taglish) Sentiment Analyzer   ")
    print("   With Lexicon Scoring, Negators, Boosters & Sarcasm ")
    print("==================================================")
    print("Commands:")
    print("  add <word> <score>      - Add lexicon word with score (-4.0 to +4.0)")
    print("  negator <word>          - Add custom negator")
    print("  booster <word> [incr]   - Add booster ('incr' or 'decr', default: incr)")
    print("  sarcasm <phrase>        - Add sarcastic phrase trigger")
    print("  quit                    - Exit program\n")

    # Run quick test suite
    print("Running initial test cases...")
    test_cases = [
        "Maganda at mabuti ang araw na ito.",
        "Hindi maganda ang lasa ng pagkain.",
        "Sobrang galing ng palabas!",
        "Wow ha, ikaw na ang magaling.",
        "Ang ganda mo, mukha kang ewan.",
        "Salamat ah, nasira mo ang kotse ko."
    ]
    for text in test_cases:
        res = tool.analyze(text)
        sarc = " [SARCASTIC]" if res["is_sarcastic"] else ""
        print(f"  Input   : '{text}'")
        print(f"  Label   : {res['label']}{sarc} (Compound: {res['adjusted_compound']})")
        if res["sarcasm_reasons"]:
            print(f"  Reasons : {', '.join(res['sarcasm_reasons'])}")
        print()

    print("Interactive Mode:")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPaalam! (Goodbye!)")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ("quit", "exit", "q"):
            print("Paalam! (Goodbye!)")
            break

        if cmd_lower.startswith("add "):
            parts = user_input.split(maxsplit=2)
            if len(parts) >= 3:
                try:
                    w, val = parts[1], float(parts[2])
                    sc = tool.add_lexicon_word(w, val)
                    print(f"[Lexicon] Added '{w}' with sentiment score {sc}")
                except ValueError:
                    print("Usage: add <word> <score from -4.0 to +4.0>")
            else:
                print("Usage: add <word> <score>")
            continue

        if cmd_lower.startswith("negator "):
            parts = user_input.split(maxsplit=1)
            if len(parts) >= 2:
                w = tool.add_negator(parts[1])
                print(f"[Negator] Added '{w}' to negators list")
            else:
                print("Usage: negator <word>")
            continue

        if cmd_lower.startswith("booster "):
            parts = user_input.split()
            if len(parts) >= 2:
                w = parts[1]
                b_type = vs.B_INCR
                if len(parts) >= 3 and parts[2].lower() in ("decr", "decrease", "down"):
                    b_type = vs.B_DECR
                word, mod = tool.add_booster(w, b_type)
                print(f"[Booster] Added '{word}' with modifier {mod}")
            else:
                print("Usage: booster <word> [incr|decr]")
            continue

        if cmd_lower.startswith("sarcasm "):
            parts = user_input.split(maxsplit=1)
            if len(parts) >= 2:
                phrase = parts[1]
                tool.add_sarcastic_phrase(phrase)
                print(f"[Sarcasm] Added sarcastic phrase trigger: '{phrase}'")
            else:
                print("Usage: sarcasm <phrase>")
            continue

        res = tool.analyze(user_input)
        raw_s = res["raw_scores"]
        print(f"  Label            : {res['label']}")
        print(f"  Adjusted Compound: {res['adjusted_compound']}")
        print(f"  Raw VADER Scores : Pos={raw_s['pos']} Neu={raw_s['neu']} Neg={raw_s['neg']} (Compound={raw_s['compound']})")
        print(f"  Sarcastic        : {res['is_sarcastic']} (Confidence: {res['sarcasm_confidence']})")
        if res["sarcasm_reasons"]:
            print(f"  Sarcasm Reasons  : {', '.join(res['sarcasm_reasons'])}")
        print()


if __name__ == "__main__":
    main()
