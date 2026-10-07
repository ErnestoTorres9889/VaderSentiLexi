import re
import math
from typing import Dict, List, Tuple, Any
import vaderSentiment.vaderSentiment as vs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SarcasmDetector:
    """Detects sarcasm and ironic contrast in Tagalog/Taglish texts."""

    def __init__(self):
        # Distinct sarcastic markers and irony/humor particles
        self.sarcastic_phrases = [
            "wow ha", "wow ah", "wow naman ha", "wow naman ah",
            "edi wow", "edi ikaw na", "sige ikaw na", "ikaw na magaling",
            "ikaw na nga magaling", "ikaw na lods", "edi shing",
            "charot", "charr", "char", "eme", "chos", "charizz", "emz",
            "wehh", "weh di nga", "pagpatuloy mo yan"
        ]

        # Structural contrast patterns indicative of sarcasm
        self.contrast_patterns = [
            # Positive praise juxtaposed with an insult/derogatory phrase
            r"\b(?:ma)?(?:ganda|galing|bait|talino|sarap|husay)\b.*?\b(?:mukha(?:ng|\s+kang)?\s+(?:ewan|tanga|bobo|sira|gago)|ewan|tanga|bobo|sira|nasira|mali|walang[ _]kwenta|basura|pangit|panget)\b",

            # Gratitude followed by damage, loss, or nuisance
            r"\bsalamat(?:\s+(?:ah|ha|din|naman))?.*?\b(?:nasira|sinira|tinapon|pinagpalit|mali|bwisit|buwisit|tapon|basura)\b",

            # Sarcastic concession: edi/sige ikaw na ...
            r"\b(?:edi|sige)\s+ikaw\s+na\b",

            # Quoted pseudo-praise: e.g. "galing", 'magaling'
            r'["\'](?:ma)?(?:galing|buti|ganda|bait|husay|lodi|petmalu)["\']'
        ]

    def add_sarcastic_phrase(self, phrase: str):
        """Add a custom sarcastic phrase trigger."""
        if not isinstance(phrase, str):
            return
        phrase_clean = phrase.strip().lower()
        if phrase_clean and phrase_clean not in self.sarcastic_phrases:
            self.sarcastic_phrases.append(phrase_clean)

    def detect(self, text: str, initial_compound: float) -> Tuple[bool, float, List[str]]:
        """Detect sarcastic intent using lexical markers, contrast patterns, and mixed polarity."""
        if not text or not isinstance(text, str):
            return False, 0.0, []

        text_lower = text.lower().strip()
        reasons = []
        score = 0.0

        # 1. Match whole-phrase sarcastic markers with word boundaries
        matched_phrases = []
        for phrase in sorted(self.sarcastic_phrases, key=len, reverse=True):
            clean_phrase = phrase.strip().lower()
            if not clean_phrase:
                continue
            pattern = rf"(?:\b|^){re.escape(clean_phrase)}(?:\b|$)"
            if re.search(pattern, text_lower):
                if not any(clean_phrase in mp for mp in matched_phrases):
                    matched_phrases.append(clean_phrase)
                    score += 0.45
                    reasons.append(f"Detected sarcastic marker: '{clean_phrase}'")

        # 2. Structural contrast patterns
        for pattern in self.contrast_patterns:
            if re.search(pattern, text_lower):
                score += 0.50
                reasons.append(f"Matched sarcastic structural contrast pattern: '{pattern}'")
                break

        # 3. Sarcastic interjection structure (e.g. 'wow ha', 'galing ah')
        interjection_pattern = r'\b(?:wow|galing|lodi|petmalu)\s+(?:naman\s+)?(?:ha|ah)\b'
        if re.search(interjection_pattern, text_lower):
            if not any("wow" in mp or "ha" in mp or "ah" in mp for mp in matched_phrases):
                score += 0.35
                reasons.append("Detected sarcastic interjection structure (e.g. 'wow ha' / 'galing ah')")

        # 4. Mixed polarity contrast: positive overall score juxtaposed with an unnegated insult,
        # excluding constructive sentences with concessive conjunctions (e.g. 'pero', 'kaso').
        is_concessive = bool(re.search(r'\b(pero|kaso|ngunit|subalit|bagamat|kahit|although|however|but)\b', text_lower))
        if initial_compound > 0.3 and not is_concessive:
            negative_words = ["ewan", "tanga", "bobo", "sira", "basura", "pangit", "panget", "kainis", "nasira", "mali"]
            for neg_word in negative_words:
                if re.search(rf'\b{re.escape(neg_word)}\b', text_lower):
                    negated = bool(re.search(rf'\b(?:hindi|di|ayaw|wala|huwag|wag)\s+{re.escape(neg_word)}\b', text_lower))
                    if not negated:
                        score += 0.35
                        reasons.append(f"Mixed polarity contrast: positive overall score with negative word '{neg_word}'")
                        break

        is_sarcastic = score >= 0.40
        confidence = min(score, 1.0)
        return is_sarcastic, round(confidence, 2), reasons


class TaglishSentiment:
    """Tagalog-English (Taglish) Sentiment Analyzer extending VADER with custom lexicons,
    negators, boosters, morphological handling, and sarcasm detection.
    """
    DEFAULT_LEXICON = {
        # Positive sentiment (Tagalog, root words, inflections, slang)
        "maganda": 2.7, "magandang": 2.7, "ganda": 2.6,
        "mabuti": 2.0, "mabuting": 2.0, "buti": 1.8,
        "magaling": 2.8, "magagaling": 2.8, "galing": 2.8,
        "masaya": 2.9, "masayang": 2.9, "saya": 2.2, "nasiyahan": 2.5,
        "mahal": 2.5, "minamahal": 2.6,
        "salamat": 2.0, "maraming salamat": 2.8,
        "astig": 2.4, "lodi": 2.5, "petmalu": 2.6,
        "masarap": 2.6, "sarap": 2.6,
        "gwapo": 2.2, "pogi": 2.2,
        "tama": 1.5, "tamang": 1.5,
        "nakakatuwa": 2.5, "natuwa": 2.3, "tuwa": 2.0,
        "ayos": 1.8, "maayos": 2.2,
        "swerte": 2.0, "swerteng": 2.0,
        "paborito": 2.4,
        "sigurado": 2.2,
        "panalo": 2.7, "panalong": 2.7, "wagi": 2.6,
        "maaasahan": 2.3,
        "respeto": 2.0,
        "sulit": 2.6, "napakasulit": 3.0,
        "mabait": 2.4, "mababait": 2.4, "bait": 2.4,
        "matalino": 2.5, "talino": 2.5,
        "husay": 2.6, "mahusay": 2.7,
        "sana all": 1.5,

        # Negative sentiment (Tagalog, root words, inflections, slang)
        "pangit": -2.7, "panget": -2.7, "kapangitan": -2.5,
        "masama": -2.5, "sama": -2.0,
        "malungkot": -2.6, "nalungkot": -2.4, "lungkot": -2.2, "nakakalungkot": -2.6,
        "galit": -2.8, "nagalit": -2.8, "nakakagalit": -2.9,
        "mahirap": -1.5, "hirap": -1.5,
        "sayang": -1.8,
        "walang kwenta": -3.0, "walang silbi": -3.0,
        "basura": -3.0,
        "nakakainis": -2.7, "nainis": -2.5, "kainis": -2.6, "inis": -2.0,
        "nakakaasar": -2.6, "naasar": -2.4, "asar": -2.2,
        "nakakasuya": -2.2, "suya": -2.0,
        "bobo": -3.0, "bobong": -3.0,
        "pagod": -1.4, "nakakapagod": -1.8,
        "takot": -2.0, "nakakatakot": -2.4,
        "sira": -2.2, "sirang": -2.2, "nasira": -2.4, "sinira": -2.5, "nakakasira": -2.4,
        "mali": -1.6, "maling": -1.6,
        "tanga": -3.2, "tangang": -3.2,
        "inutil": -3.1,
        "kupal": -3.5,
        "epal": -2.5,
        "pabida": -2.4,
        "ewan": -1.2,
        "bwisit": -2.8, "buwisit": -2.8, "bwiset": -2.8,
        "saklap": -2.5, "lugi": -2.2,
        "bastos": -2.8, "salbahe": -2.6, "kadiri": -2.8,
    }

    DEFAULT_NEGATORS = [
        "hindi", "di", "hinde", "ayaw", "ayoko", "wala", "walang",
        "huwag", "wag", "hwag", "hinding-hindi", "dili"
    ]

    DEFAULT_BOOSTERS = {
        "sobrang": vs.B_INCR,
        "sobra": vs.B_INCR,
        "napaka": vs.B_INCR,
        "grabe": vs.B_INCR,
        "grabeng": vs.B_INCR,
        "talaga": vs.B_INCR,
        "talagang": vs.B_INCR,
        "ubod": vs.B_INCR,
        "ubod ng": vs.B_INCR,
        "masyadong": vs.B_INCR,
        "masyado": vs.B_INCR,
        "lalo": vs.B_INCR,
        "lalong": vs.B_INCR,
        "tunay": vs.B_INCR,
        "tunay na": vs.B_INCR,
        "medyo": vs.B_DECR,
        "konti": vs.B_DECR,
        "kaunti": vs.B_DECR,
        "bahagya": vs.B_DECR,
        "bahagyang": vs.B_DECR,
    }

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sarcasm_detector = SarcasmDetector()
        self._multiword_lexicon = set()
        self._initialize_vader_extension()

    def _initialize_vader_extension(self):
        """Extend VADER analyzer with initial Tagalog lexicons, negators, and boosters."""
        for word, score in self.DEFAULT_LEXICON.items():
            self._register_lexicon_entry(word, score)

        for word in self.DEFAULT_NEGATORS:
            if word not in vs.NEGATE:
                vs.NEGATE.append(word)

        vs.BOOSTER_DICT.update(self.DEFAULT_BOOSTERS)

    def _register_lexicon_entry(self, word: str, score: float):
        """Register a word/phrase in lexicon, supporting multi-word expressions via underscores."""
        clean_word = word.strip().lower()
        if not clean_word:
            return
        self.analyzer.lexicon[clean_word] = score
        if " " in clean_word:
            self._multiword_lexicon.add(clean_word)
            underscored = clean_word.replace(" ", "_")
            self.analyzer.lexicon[underscored] = score

    def add_lexicon_word(self, word: str, score: float) -> float:
        """Add or update a lexicon word with a sentiment score (-4.0 to 4.0)."""
        if not isinstance(word, str):
            raise ValueError("Word must be a string")
        clean_word = word.strip().lower()
        if not clean_word:
            raise ValueError("Word cannot be empty")
        try:
            val = float(score)
            if math.isnan(val):
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError("Score must be a valid float number")

        bounded_score = max(-4.0, min(4.0, val))
        self._register_lexicon_entry(clean_word, bounded_score)
        return bounded_score

    def remove_lexicon_word(self, word: str) -> bool:
        """Remove a word or phrase from the active lexicon."""
        if not isinstance(word, str):
            return False
        clean_word = word.strip().lower()
        found = False
        if clean_word in self.analyzer.lexicon:
            del self.analyzer.lexicon[clean_word]
            found = True
        if " " in clean_word:
            self._multiword_lexicon.discard(clean_word)
            underscored = clean_word.replace(" ", "_")
            if underscored in self.analyzer.lexicon:
                del self.analyzer.lexicon[underscored]
                found = True
        return found

    def add_negator(self, word: str) -> str:
        """Add a custom negator word."""
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Negator must be a non-empty string")
        clean_word = word.strip().lower()
        if clean_word not in vs.NEGATE:
            vs.NEGATE.append(clean_word)
        return clean_word

    def add_booster(self, word: str, modifier: float = vs.B_INCR) -> Tuple[str, float]:
        """Add a custom booster / intensifier word with modifier value."""
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Booster word must be a non-empty string")
        clean_word = word.strip().lower()
        try:
            val = float(modifier)
            if math.isnan(val):
                val = vs.B_INCR
        except (ValueError, TypeError):
            val = vs.B_INCR
        vs.BOOSTER_DICT[clean_word] = val
        return clean_word, val

    def add_sarcastic_phrase(self, phrase: str):
        """Add a custom sarcastic phrase trigger."""
        self.sarcasm_detector.add_sarcastic_phrase(phrase)

    @staticmethod
    def _clean(text: str) -> str:
        """Clean and normalize input text while preserving casing for emphasis."""
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return re.sub(r"\s+", " ", text.strip())

    def _prepare_text_for_vader(self, text: str) -> str:
        """Preprocess text for VADER by decomposing superlative affixes and joining multi-word idioms."""
        # 1. Expand superlative prefix 'napaka-' (e.g. 'napakaganda' -> 'napaka ganda')
        expanded = re.sub(r'\bnapaka([a-z]{3,})\b', r'napaka \1', text, flags=re.IGNORECASE)

        # 2. Join multi-word lexicon phrases with underscores so VADER tokenizes them as single tokens
        sorted_phrases = sorted(self._multiword_lexicon, key=len, reverse=True)
        for phrase in sorted_phrases:
            pattern = rf'\b{re.escape(phrase)}\b'
            expanded = re.sub(pattern, phrase.replace(' ', '_'), expanded, flags=re.IGNORECASE)

        return expanded

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment and structural sarcasm.

        Returns detailed dictionary containing:
        - raw sentiment scores
        - sarcasm detection results
        - adjusted compound score
        - final sentiment label
        """
        cleaned_text = self._clean(text)
        if not cleaned_text:
            return {
                "text": "",
                "raw_scores": {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0},
                "adjusted_compound": 0.0,
                "is_sarcastic": False,
                "sarcasm_confidence": 0.0,
                "sarcasm_reasons": [],
                "label": "NEUTRAL"
            }

        vader_input = self._prepare_text_for_vader(cleaned_text)
        scores = self.analyzer.polarity_scores(vader_input)
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
                adjusted_compound = min(raw_compound - 0.25, -0.45)

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
    print("  add <word> <score>      - Add lexicon word/phrase with score (-4.0 to +4.0)")
    print("  negator <word>          - Add custom negator")
    print("  booster <word> [incr]   - Add booster ('incr', 'decr', or float, default: incr)")
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
        print(f"  Input   : '{text}'")
        print(f"  Label   : {res['label']} (Compound: {res['adjusted_compound']})")
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
            arg_str = user_input[4:].strip()
            parts = arg_str.rsplit(maxsplit=1)
            if len(parts) == 2:
                try:
                    w, val = parts[0], float(parts[1])
                    sc = tool.add_lexicon_word(w, val)
                    print(f"[Lexicon] Added '{w}' with sentiment score {sc}")
                except ValueError:
                    print("Usage: add <word/phrase> <score from -4.0 to +4.0>")
            else:
                print("Usage: add <word/phrase> <score>")
            continue

        if cmd_lower.startswith("negator "):
            word_to_add = user_input[8:].strip()
            if word_to_add:
                try:
                    w = tool.add_negator(word_to_add)
                    print(f"[Negator] Added '{w}' to negators list")
                except ValueError as e:
                    print(f"[Error] {e}")
            else:
                print("Usage: negator <word>")
            continue

        if cmd_lower.startswith("booster "):
            arg_str = user_input[8:].strip()
            parts = arg_str.rsplit(maxsplit=1) if " " in arg_str else [arg_str]
            if parts and parts[0]:
                b_word = parts[0]
                b_type = vs.B_INCR
                if len(parts) == 2:
                    flag = parts[1].lower()
                    if flag in ("decr", "decrease", "down"):
                        b_type = vs.B_DECR
                    elif flag in ("incr", "increase", "up"):
                        b_type = vs.B_INCR
                    else:
                        try:
                            b_type = float(parts[1])
                        except ValueError:
                            b_type = vs.B_INCR
                try:
                    word, mod = tool.add_booster(b_word, b_type)
                    print(f"[Booster] Added '{word}' with modifier {mod}")
                except ValueError as e:
                    print(f"[Error] {e}")
            else:
                print("Usage: booster <word> [incr|decr|<float>]")
            continue

        if cmd_lower.startswith("sarcasm "):
            phrase_to_add = user_input[8:].strip()
            if phrase_to_add:
                tool.add_sarcastic_phrase(phrase_to_add)
                print(f"[Sarcasm] Added sarcastic phrase trigger: '{phrase_to_add}'")
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
