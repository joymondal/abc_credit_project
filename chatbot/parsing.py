"""Deterministic parsers and validators for free-text answers (the validation layer)."""
from __future__ import annotations

import re

_UNITS = {"k": 1e3, "thousand": 1e3, "l": 1e5, "lac": 1e5, "lacs": 1e5, "lakh": 1e5, "lakhs": 1e5,
          "cr": 1e7, "crore": 1e7}
YES = {"yes", "y", "yeah", "yep", "ya", "haan", "ha", "ok", "okay", "sure", "correct", "right",
       "confirm", "submit", "agree", "i agree", "true"}
NO = {"no", "n", "nope", "nah", "nahi", "none", "not", "false", "disagree", "skip"}


def inr(x) -> str:
    """85000 -> 'Rs 85,000' with Indian digit grouping."""
    n = int(round(float(x)))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return ("-" if n < 0 else "") + "Rs " + s


def parse_amount(text):
    """'25k' -> 25000, '1.2 lakh' -> 120000, 'Rs 45,000/-' -> 45000, '₹ 90000' -> 90000."""
    if text is None:
        return None
    s = str(text).lower().replace(",", "").replace("₹", " ").replace("/-", " ")
    s = re.sub(r"\b(rs\.?|inr|rupees?)\b", " ", s)
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|l|thousand|k|crores?|cr)?\b", s)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2):
        val *= _UNITS[m.group(2).rstrip("s") if m.group(2) not in _UNITS else m.group(2)]
    return int(round(val))


def parse_int(text):
    m = re.search(r"(?<![\d.])(\d{1,3})(?![\d.])", str(text or ""))
    return int(m.group(1)) if m else None


def parse_pincode(text):
    m = re.search(r"(?<!\d)([1-9]\d{5})(?!\d)", str(text or "").replace(" ", ""))
    if not m:                                  # allow '560 037'
        m = re.search(r"(?<!\d)([1-9]\d{2})\s?(\d{3})(?!\d)", str(text or ""))
        return int(m.group(1) + m.group(2)) if m else None
    return int(m.group(1))


def parse_yes_no(text):
    words = re.findall(r"[a-z0-9']+", str(text or "").lower())
    if not words:
        return None
    if words[0] in NO or " ".join(words[:2]) in {"do not", "dont have", "don't have"}:
        return False
    if words[0] in YES or " ".join(words[:2]) in YES:
        return True
    return None


def _tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def match_option(text, options: dict):
    """Match free text / a number to one of `options` = {code: (label, [synonyms...])}."""
    t = str(text or "").strip().lower()
    codes = list(options)
    if re.fullmatch(r"\d{1,2}", t) and 1 <= int(t) <= len(codes):
        return codes[int(t) - 1]
    toks = set(_tokens(t))
    best, best_score = None, 0
    for code, (label, syn) in options.items():
        names = [code.lower(), label.lower(), *[s.lower() for s in syn]]
        score = 0
        for nme in names:
            if t == nme:
                score = max(score, 3)
            elif nme in t and len(nme) > 2:      # phrase containment ('salaried employee')
                score = max(score, 2 + len(nme) / 100)
            elif toks & set(_tokens(nme)):
                score = max(score, 1)
        if score > best_score:
            best, best_score = code, score
    return best if best_score >= 2 else None


def match_vehicle(text, catalogue: dict, k: int = 6):
    """Fuzzy match a typed vehicle name against the catalogue.

    Returns (exact_key or None, [candidate keys best-first]).
    Score = share of query tokens found in the model description (prefix match allowed),
    ties broken by popularity. Deterministic - no language model involved.
    """
    q = str(text or "").upper().strip()
    if q in catalogue:
        return q, [q]
    stop = {"tvs", "bsvi", "bs6", "bs", "vi", "the", "a", "i", "want", "to", "buy", "bike", "scooter",
            "model", "new", "for", "finance", "loan", "my", "is", "it", "and", "with", "cc"}
    qt = [t for t in _tokens(q) if t not in stop]
    if not qt:
        return None, []
    scored = []
    for key, v in catalogue.items():
        dt = _tokens(key + " " + v.get("Make_Code", "") + " " + v.get("alias", ""))
        hit = sum(1 for t in qt if any(d == t or (len(t) >= 3 and d.startswith(t)) for d in dt))
        if hit:
            scored.append((hit / len(qt), v.get("n", 0), key))
    scored.sort(key=lambda r: (-r[0], -r[1]))
    if not scored or scored[0][0] < 0.5:
        return None, []
    top = [s for s in scored if s[0] == scored[0][0]]
    if len(top) == 1 and top[0][0] == 1.0:
        return top[0][2], [top[0][2]]
    return None, [s[2] for s in scored[:k]]
