import os, sys, re, json
from docx import Document

# ---------- filename meta ----------

def parse_meta(fname):
    """
    Parse '<NN>_ptm_<id>.docx' or '<NN>_ptv_<id>.docx'.
    Returns (knesset:int, proto_type:str) or (None, None) if bad.
    """
    m = re.fullmatch(r"(\d+)_pt([mv])_(\d+)\.docx", fname)
    if not m:
        return None, None
    return int(m.group(1)), ("plenary" if m.group(2) == "m" else "committee")

# ---------- chair & protocol number ----------

def extract_protocol_number(pars):
    """
    Find protocol/meeting number. If missing -> -1.
    """
    pats = [
        r'פרוטוקול\s*מס[\'"]?\s*(\d+)',
        r'מספר\s*ישיבה[:\s]+(\d+)',
        r'ישיבה\s*מס[\'"]?\s*(\d+)',
        r'פרוטוקול[:\s]+(\d+)',
    ]
    for p in pars:
        s = p.strip()
        for pat in pats:
            m = re.search(pat, s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    return -1

_ROLE_WORDS = {
    'היו"ר','יו"ר','יושב-ראש','יושב','ראש','מ"מ','ח"כ','עו"ד','ד"ר','פרופ',
    'גב׳','גב\'','גברת','מר','מר.','שר','שרה','השר','השרה','סגן','סגנית',
    'ראש','הממשלה','מנכ"ל','מנכ״ל'
}

_NON_SPEAKER_HEADERS = {
    "מסקנות הוועדה","מנהלות הוועדה","מנהלת הוועדה","רשמת פרלמנטרית",
    "רישום פרלמנטרי","קריאות","קריאה","נכחו","חברי הכנסת",
    "רשימת המוזמנים","סדר היום","הישיבה נפתחה","הישיבה ננעלה","פרוטוקול",
    "מסמכים שהונחו על שולחן הכנסת"
}

def clean_name(txt):
    """Remove roles, parentheses, extra spaces; keep only Hebrew name-like text."""
    if not txt: 
        return ""
    s = txt.strip().strip(":")
    s = re.sub(r"\(.*?\)", "", s)
    # remove role words as whole words
    if _ROLE_WORDS:
        pat = r"\b(" + "|".join(re.escape(w) for w in _ROLE_WORDS) + r")\b"
        s = re.sub(pat, "", s)
    s = re.sub(r"[:\-\u2013\u2014]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    # must contain Hebrew and look short (1-4 tokens)
    if not re.search(r"[א-ת]", s): 
        return ""
    toks = s.split()
    if not (1 <= len(toks) <= 4): 
        return ""
    # reject quoted/agenda-like starts
    if s[:1] in {'"', '“', '”', '״', "'", "׳", '('}:
        return ""
    if s in _NON_SPEAKER_HEADERS:
        return ""
    return s

def detect_chair(pars):
    """
    Look for explicit 'יו\"ר/היו\"ר/יושב-ראש: <name>'.
    Return "" if not found.
    """
    pat = r'(?:יו"?ר|היו"?ר|יושב[\-\s]ראש)[^:]{0,20}:\s*([א-ת"״׳\'\-\s]+)'
    for p in pars:
        m = re.search(pat, p)
        if m:
            cand = clean_name(m.group(1))
            if cand:
                return cand
    return ""

# ---------- speaker tags & headers ----------

def looks_like_header(line):
    """
    Agenda/opening lines that shouldn't become sentences or speakers.
    """
    t = line.strip().strip(":").strip()
    if not t or len(t) <= 2: 
        return True
    if t in _NON_SPEAKER_HEADERS: 
        return True
    if re.match(r"^(סדר היום|רשימת המוזמנים|נכחו|קריאות)\b", t): 
        return True
    if re.match(r"^(הישיבה\b|ישיבה\s+\d+)", t): 
        return True
    if re.match(r"^יום\s+[א-ת0-9\"׳״\-\(\)]", t): 
        return True
    if re.match(r"^ירושלים\b", t) or re.match(r"^שעה\b", t): 
        return True
    if re.match(r"^(סקירת|הודעת|מסמכים\s+שהונחו|תוכנ(?:ית|ה)|ברכת|דרכים)\b", t): 
        return True
    if re.search(r"\bהצעת\s+חוק\b", t) or re.search(r"\bקריאה\s+(?:ראשונה|שנייה|שלישית)\b", t):
        return True
    return False

def speaker_from_line(line):
    """
    Accept '<name>:' after cleaning, reject headers/quotes.
    """
    s = line.strip()
    if not s.endswith(":"): 
        return ""
    cand = clean_name(s[:-1])
    if not cand or looks_like_header(cand): 
        return ""
    return cand

# ---------- sentence split & tokenize ----------

_ABBR_QUOTE = re.compile(r'([א-ת])["״׳\']([א-ת])')  # keep things like ח"כ as one

def split_sentences(text):
    """Split on . ? ! ; (not on colon)."""
    t = text.replace("…", ".")
    parts = re.split(r'(?<=[\.\?\!;])\s+', t)
    return [p.strip() for p in parts if p.strip()]

def tokenize(sent):
    """
    Basic Hebrew tokenization:
    - keep ח"כ/יו"ר intact
    - keep hyphen inside Hebrew word
    - split general punctuation to separate tokens
    """
    s = sent.strip()
    s = _ABBR_QUOTE.sub(r"\1<QQ>\2", s)              # protect quotes inside Hebrew word
    s = re.sub(r'([א-ת])\-([א-ת])', r'\1<HH>\2', s)  # protect inner hyphen
    s = re.sub(r'([\.\,\!\?\;\:\(\)\[\]\{\}"“”\'׳״/])', r' \1 ', s)
    s = re.sub(r'\-', ' - ', s)
    s = s.replace("<HH>", "-").replace("<QQ>", '"')
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if t]
    return toks

def heb_ratio(s):
    """Share of Hebrew letters in string."""
    if not s: 
        return 0.0
    return len(re.findall(r"[א-ת]", s)) / max(1, len(s))

def normalize_sentence(sent):
    """
    Drop obvious noise and require >=4 tokens with some Hebrew.
    Use a mild filter (to keep more data).
    """
    s = re.sub(r"[_]{2,}", " ", sent)
    s = re.sub(r"\-\s*\-\s*\-+", " ", s)
    s = s.replace("•", " ")
    if heb_ratio(s) < 0.25:   # not too strict
        return ""
    toks = tokenize(s)
    if len(toks) < 4:
        return ""
    return " ".join(toks)

# ---------- core per document ----------

def process_doc(path):
    """
    Read one .docx and return list of JSONL-ready dicts.
    """
    out = []
    fname = os.path.basename(path)
    knes, ptype = parse_meta(fname)
    if knes is None: 
        return out
    try:
        doc = Document(path)
    except Exception:
        return out
    pars = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    pnum = extract_protocol_number(pars)
    chair = ""

    current = ""
    seen_speaker = False

    for raw in pars:
        # new speaker?
        sp = speaker_from_line(raw)
        if sp:
            current = sp
            seen_speaker = True
            # update chair if tag shows יו"ר and we don't have one yet
            if not chair and re.search(r'(?:יו"?ר|היו"?ר|יושב[\-\s]ראש)', raw):
                chair = current
            continue

        # skip global headers
        if looks_like_header(raw):
            continue

        # avoid attributing pre-speech lines
        if not seen_speaker or not current:
            continue

        # push sentences
        for s in split_sentences(raw):
            norm = normalize_sentence(s)
            if not norm:
                continue
            out.append({
                "protocol_name": fname,
                "knesset_number": int(knes),
                "protocol_type": ptype,                 # "committee" | "plenary"
                "protocol_number": int(pnum),           # -1 if missing
                "protocol_chairmain": chair or "",      # empty string if unknown
                "speaker_name": current,
                "sentence_text": norm
            })
    return out

# ---------- directory & I/O ----------

def process_dir(indir, out_jsonl):
    all_rows = []
    try:
        for name in os.listdir(indir):
            if name.lower().endswith(".docx"):
                rows = process_doc(os.path.join(indir, name))
                if rows:
                    all_rows.extend(rows)
    except Exception:
        return 3  # read/scan error
    try:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return 4  # write error
    return 0      # ok

# ---------- CLI ----------

def main(argv):
    # expects: input_dir, output.jsonl
    if len(argv) != 3:
        return 1
    if not os.path.isdir(argv[1]):
        return 2
    return process_dir(argv[1], argv[2])

if __name__ == "__main__":
    sys.exit(main(sys.argv))
