import re
import math
import os
from typing import Dict, List, Tuple, Any


class SentimentIntensityAnalyzer:

    B_INCR = 0.293
    B_DECR = -0.293
    C_INCR = 0.733
    N_SCALAR = -0.74

    def __init__(self, lexicon_file: str = "Vader.txt"):
        self.lexicon: Dict[str, float] = {}
        self.negate: List[str] = [
            "hindi", "di", "hinde", "ayaw", "ayoko", "wala", "walang",
            "huwag", "wag", "hwag", "hinding-hindi", "dili",
            "not", "no", "never", "without", "neither", "nor", "none"
        ]
        self.booster_dict: Dict[str, float] = {
            "sobrang": self.B_INCR, "sobra": self.B_INCR, "napaka": self.B_INCR,
            "grabe": self.B_INCR, "grabeng": self.B_INCR, "talaga": self.B_INCR,
            "talagang": self.B_INCR, "ubod": self.B_INCR, "ubod ng": self.B_INCR,
            "masyadong": self.B_INCR, "masyado": self.B_INCR, "lalo": self.B_INCR,
            "lalong": self.B_INCR, "tunay": self.B_INCR, "tunay na": self.B_INCR,
            "medyo": self.B_DECR, "konti": self.B_DECR, "kaunti": self.B_DECR,
            "bahagya": self.B_DECR, "bahagyang": self.B_DECR,
            "very": self.B_INCR, "extremely": self.B_INCR, "really": self.B_INCR,
            "so": self.B_INCR, "too": self.B_INCR, "super": self.B_INCR,
            "slightly": self.B_DECR, "somewhat": self.B_DECR, "kind of": self.B_DECR
        }
        self.lexicon_file = lexicon_file
        if lexicon_file:
            self.load_lexicon_from_file(lexicon_file)

    def load_lexicon_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.rsplit(maxsplit=1)
                if len(parts) == 2:
                    word, score_str = parts[0].strip().lower(), parts[1].strip()
                    try:
                        score = float(score_str)
                        self.lexicon[word] = score
                        if " " in word:
                            self.lexicon[word.replace(" ", "_")] = score
                    except ValueError:
                        continue

    def polarity_scores(self, text: str) -> Dict[str, float]:
        if not text or not isinstance(text, str):
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

        cleaned_text = text.strip()
        words = re.findall(r"\b[\w'-]+\b", cleaned_text)
        if not words:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

        exclamation_count = min(text.count("!"), 4)
        punct_amplifier = exclamation_count * 0.292

        has_lower = any(c.islower() for c in text)
        has_upper = any(c.isupper() for c in text)
        is_mixed_case = has_lower and has_upper

        sentiments = []
        words_lower = [w.lower() for w in words]

        for i, word in enumerate(words):
            word_lower = words_lower[i]
            if word_lower not in self.lexicon:
                continue

            score = self.lexicon[word_lower]

            if word.isupper() and is_mixed_case:
                if score > 0:
                    score += self.C_INCR
                else:
                    score -= self.C_INCR

            scalar = 0.0
            is_negated = False
            for k in range(1, 4):
                if i - k >= 0:
                    prev_word = words_lower[i - k]
                    if prev_word in self.booster_dict:
                        b_val = self.booster_dict[prev_word]
                        if words[i - k].isupper() and is_mixed_case:
                            b_val += self.C_INCR if b_val > 0 else -self.C_INCR
                        scalar += b_val
                    if prev_word in self.negate:
                        is_negated = True

            if score > 0:
                score += scalar
            elif score < 0:
                score -= scalar

            if is_negated:
                score *= self.N_SCALAR

            sentiments.append(score)

        sum_s = sum(sentiments)
        if sum_s > 0:
            sum_s += punct_amplifier
        elif sum_s < 0:
            sum_s -= punct_amplifier

        compound = sum_s / math.sqrt(sum_s**2 + 15.0) if sum_s != 0 else 0.0

        pos_sum = sum(s for s in sentiments if s > 0)
        neg_sum = sum(abs(s) for s in sentiments if s < 0)

        if sum_s > 0:
            pos_sum += punct_amplifier
        elif sum_s < 0:
            neg_sum += punct_amplifier

        neu_count = len(words) - len(sentiments)
        total_score = pos_sum + neg_sum + neu_count

        if total_score > 0:
            pos = round(pos_sum / total_score, 3)
            neg = round(neg_sum / total_score, 3)
            neu = round(neu_count / total_score, 3)
        else:
            pos = neg = 0.0
            neu = 1.0

        return {
            "neg": neg,
            "neu": neu,
            "pos": pos,
            "compound": round(compound, 4)
        }


class SarcasmDetector:

    def __init__(self):
        self.sarcastic_phrases = [
            "wow ha", "wow ah", "wow naman ha", "wow naman ah",
            "edi wow", "edi ikaw na", "sige ikaw na", "ikaw na magaling",
            "ikaw na nga magaling", "ikaw na lods", "edi shing",
            "charot", "charr", "char", "eme", "chos", "charizz", "emz",
            "wehh", "weh di nga", "pagpatuloy mo yan"
        ]

        self.contrast_patterns = [
            r"\b(?:ma)?(?:ganda|galing|bait|talino|sarap|husay)\b.*?\b(?:mukha(?:ng|\s+kang)?\s+(?:ewan|tanga|bobo|sira|gago)|ewan|tanga|bobo|sira|nasira|mali|walang[ _]kwenta|basura|pangit|panget)\b",
            r"\bsalamat(?:\s+(?:ah|ha|din|naman))?.*?\b(?:nasira|sinira|tinapon|pinagpalit|mali|bwisit|buwisit|tapon|basura)\b",
            r"\b(?:edi|sige)\s+ikaw\s+na\b",
            r'["\'](?:ma)?(?:galing|buti|ganda|bait|husay|lodi|petmalu)["\']'
        ]

    def add_sarcastic_phrase(self, phrase: str):
        if not isinstance(phrase, str):
            return
        phrase_clean = phrase.strip().lower()
        if phrase_clean and phrase_clean not in self.sarcastic_phrases:
            self.sarcastic_phrases.append(phrase_clean)

    def remove_sarcastic_phrase(self, phrase: str) -> bool:
        if not isinstance(phrase, str):
            return False
        phrase_clean = phrase.strip().lower()
        if phrase_clean in self.sarcastic_phrases:
            self.sarcastic_phrases.remove(phrase_clean)
            return True
        return False

    def detect(self, text: str, initial_compound: float) -> Tuple[bool, float, List[str]]:
        if not text or not isinstance(text, str):
            return False, 0.0, []

        text_lower = text.lower().strip()
        reasons = []
        score = 0.0

        matched_phrases = []
        for phrase in sorted(self.sarcastic_phrases, key=len, reverse=True):
            clean_phrase = phrase.strip().lower()
            if not clean_phrase:
                continue
            prefix = r"(?:\b|^)" if clean_phrase[0].isalnum() or clean_phrase[0] == '_' else r"(?:(?<=\s)|^)"
            suffix = r"(?:\b|$)" if clean_phrase[-1].isalnum() or clean_phrase[-1] == '_' else r"(?:(?=\s)|$)"
            pattern = f"{prefix}{re.escape(clean_phrase)}{suffix}"
            if re.search(pattern, text_lower):
                if not any(clean_phrase in mp for mp in matched_phrases):
                    matched_phrases.append(clean_phrase)
                    score += 0.45
                    reasons.append(f"Detected sarcastic marker: '{clean_phrase}'")

        for pattern in self.contrast_patterns:
            if re.search(pattern, text_lower):
                score += 0.50
                reasons.append(f"Matched sarcastic structural contrast pattern: '{pattern}'")
                break

        interjection_pattern = r'\b(?:wow|galing|lodi|petmalu)\s+(?:naman\s+)?(?:ha|ah)\b'
        if re.search(interjection_pattern, text_lower):
            if not any("wow" in mp or "ha" in mp or "ah" in mp for mp in matched_phrases):
                score += 0.35
                reasons.append("Detected sarcastic interjection structure (e.g. 'wow ha' / 'galing ah')")

        is_concessive = bool(re.search(r'\b(pero|kaso|ngunit|subalit|bagamat|kahit|although|however|but)\b', text_lower))
        if initial_compound > 0.3 and not is_concessive:
            negative_words = ["ewan", "tanga", "bobo", "sira", "basura", "pangit", "panget", "kainis", "nasira", "mali"]
            for neg_word in negative_words:
                if re.search(rf'\b{re.escape(neg_word)}\b', text_lower):
                    negated = bool(re.search(rf'\b(?:hindi|di|ayaw|wala|huwag|wag)\s+(?:\w+\s+)?{re.escape(neg_word)}\b', text_lower))
                    if not negated:
                        score += 0.35
                        reasons.append(f"Mixed polarity contrast: positive overall score with negative word '{neg_word}'")
                        break

        is_sarcastic = score >= 0.40
        confidence = min(score, 1.0)
        return is_sarcastic, round(confidence, 2), reasons


class TaglishSentiment:

    def __init__(self, lexicon_file: str = "Vader.txt"):
        self.lexicon_file = lexicon_file
        self.analyzer = SentimentIntensityAnalyzer(lexicon_file=lexicon_file)
        self.sarcasm_detector = SarcasmDetector()
        self._multiword_lexicon = set()
        self._update_multiword_set()

    def _update_multiword_set(self):
        self._multiword_lexicon = {w for w in self.analyzer.lexicon if " " in w}

    def _register_lexicon_entry(self, word: str, score: float):
        clean_word = word.strip().lower()
        if not clean_word:
            return
        self.analyzer.lexicon[clean_word] = score
        if " " in clean_word:
            self._multiword_lexicon.add(clean_word)
            underscored = clean_word.replace(" ", "_")
            self.analyzer.lexicon[underscored] = score

    def add_lexicon_word(self, word: str, score: float) -> float:
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
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Negator must be a non-empty string")
        clean_word = word.strip().lower()
        if clean_word not in self.analyzer.negate:
            self.analyzer.negate.append(clean_word)
        return clean_word

    def add_booster(self, word: str, modifier: float = SentimentIntensityAnalyzer.B_INCR) -> Tuple[str, float]:
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Booster word must be a non-empty string")
        clean_word = word.strip().lower()
        try:
            val = float(modifier)
            if math.isnan(val):
                val = SentimentIntensityAnalyzer.B_INCR
        except (ValueError, TypeError):
            val = SentimentIntensityAnalyzer.B_INCR
        self.analyzer.booster_dict[clean_word] = val
        return clean_word, val

    def add_sarcastic_phrase(self, phrase: str):
        self.sarcasm_detector.add_sarcastic_phrase(phrase)

    def remove_negator(self, word: str) -> bool:
        if not isinstance(word, str):
            return False
        clean_word = word.strip().lower()
        if clean_word in self.analyzer.negate:
            self.analyzer.negate.remove(clean_word)
            return True
        return False

    def remove_booster(self, word: str) -> bool:
        if not isinstance(word, str):
            return False
        clean_word = word.strip().lower()
        if clean_word in self.analyzer.booster_dict:
            del self.analyzer.booster_dict[clean_word]
            return True
        return False

    def remove_sarcastic_phrase(self, phrase: str) -> bool:
        return self.sarcasm_detector.remove_sarcastic_phrase(phrase)

    def remove_item(self, item: str, category: str = "any") -> Dict[str, bool]:
        if not isinstance(item, str) or not item.strip():
            return {"lexicon": False, "negator": False, "booster": False, "sarcasm": False}
        clean_item = item.strip().lower()
        category = (category or "any").strip().lower()

        results = {
            "lexicon": False,
            "negator": False,
            "booster": False,
            "sarcasm": False
        }

        if category in ("lexicon", "word", "any"):
            results["lexicon"] = self.remove_lexicon_word(clean_item)
        if category in ("negator", "neg", "any"):
            results["negator"] = self.remove_negator(clean_item)
        if category in ("booster", "boost", "any"):
            results["booster"] = self.remove_booster(clean_item)
        if category in ("sarcasm", "sarcastic", "phrase", "any"):
            results["sarcasm"] = self.remove_sarcastic_phrase(clean_item)

        return results

    @staticmethod
    def _clean(text: str) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return re.sub(r"\s+", " ", text.strip())

    def _prepare_text_for_vader(self, text: str) -> str:
        expanded = re.sub(r'\bnapaka-?([a-z]{3,})\b', r'napaka \1', text, flags=re.IGNORECASE)

        sorted_phrases = sorted(self._multiword_lexicon, key=len, reverse=True)
        for phrase in sorted_phrases:
            prefix = r'\b' if phrase[0].isalnum() or phrase[0] == '_' else r'(?:(?<=\s)|^)'
            suffix = r'\b' if phrase[-1].isalnum() or phrase[-1] == '_' else r'(?:(?=\s)|$)'
            pattern = f'{prefix}{re.escape(phrase)}{suffix}'
            expanded = re.sub(pattern, phrase.replace(' ', '_'), expanded, flags=re.IGNORECASE)

        return expanded

    def analyze(self, text: str) -> Dict[str, Any]:
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

        is_sarcastic, sarcasm_conf, reasons = self.sarcasm_detector.detect(cleaned_text, raw_compound)

        adjusted_compound = raw_compound
        if is_sarcastic:
            if raw_compound > 0:
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
    print("      (Using Vader.txt Lexicon & Pure Python)     ")
    print("==================================================")
    print("Commands:")
    print("  add <word> <score>               - Add lexicon word/phrase with score (-4.0 to +4.0)")
    print("  negator <word>                   - Add custom negator")
    print("  booster <word> [incr]            - Add booster ('incr', 'decr', or float, default: incr)")
    print("  sarcasm <phrase>                 - Add sarcastic phrase trigger")
    print("  remove [type] <word/phrase>      - Remove item (type: 'lexicon', 'negator', 'booster', 'sarcasm', or omitted)")
    print("  quit                             - Exit program\n")

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
                b_type = SentimentIntensityAnalyzer.B_INCR
                if len(parts) == 2:
                    flag = parts[1].lower()
                    if flag in ("decr", "decrease", "down"):
                        b_type = SentimentIntensityAnalyzer.B_DECR
                    elif flag in ("incr", "increase", "up"):
                        b_type = SentimentIntensityAnalyzer.B_INCR
                    else:
                        try:
                            b_type = float(parts[1])
                        except ValueError:
                            b_type = SentimentIntensityAnalyzer.B_INCR
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

        if cmd_lower.startswith("remove ") or cmd_lower == "remove":
            arg_str = user_input[6:].strip() if cmd_lower.startswith("remove ") else ""
            if not arg_str:
                print("Usage: remove [lexicon|negator|booster|sarcasm] <word/phrase>  OR  remove <word/phrase>")
                continue

            parts = arg_str.split(maxsplit=1)
            target_cat = "any"
            target_item = arg_str

            if len(parts) == 2 and parts[0].lower() in ("lexicon", "word", "negator", "booster", "sarcasm", "sarcastic", "phrase"):
                target_cat = parts[0].lower()
                target_item = parts[1]

            res_map = tool.remove_item(target_item, target_cat)
            removed_from = [cat for cat, removed in res_map.items() if removed]

            if removed_from:
                print(f"[Remove] Removed '{target_item}' from: {', '.join(removed_from)}")
            else:
                print(f"[Remove] Item '{target_item}' was not found in specified category ({target_cat})")
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
