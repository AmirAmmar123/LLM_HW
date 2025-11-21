import re
import os
from docx import Document
from multiprocessing import Pool, cpu_count
import json

PATH = r"./to_txt"

# INVALID TALKERS
INVALID_TALKERS_NAMES = {
    'חברי כנסת','משתתפים באמצעים מקוונים','מנהלות הוועדה','מסקנות הוועדה',
    "חברי הוועדה","מוזמנים","קריאה","אני מבקש לשאול",'נכחו','סדר היום',
    'חברי הכנסת','באמצעים מקוונים','ייעוץ משפטי','רישום פרלמנטרי','',
}

# SUFFIXES TO REMOVE
GET_RID_OF_SUFFIX = {
    'יו"ר ועדת הכנסת','שר החינוך','היו"ר','היו”ר','לאיכות הסביבה',
    'השר לאיכות הסביבה','שר הבריאות','סגן שר החינוך, התרבות והספורט',
    'שר החינוך, התרבות והספורט','שר המשפטים','רצוני לשאול','שר האוצר',
    'שרת העבודה והרווחה','שר התחבורה','תשובת שר התחבורה','שר המשטרה',
    'תשובת שר המשטרה','שר החקלאות','חבר הכנסת','שאל את ביום','משרד המשפטים',
    'שר התיירות','סגן מזכיר הכנסת','השר','שר התקשורת','השר לענייני דתות',
    'מזכיר הכנסת','מנהלת הוועדה','שר הפנים','תשובת','להגנת הסביבה',
    'השר להגנת הסביבה','שר התשתיות הלאומיות','שר הביטחון','שר החוץ',
    'שר האנרגיה','שר האנרגיה והתשתיות','שר המדע','שר הרווחה','שר הכלכלה',
    'שר הקליטה','שר הדתות','שר לביטחון הפנים','השרה להגנת הסביבה',
    'השרה לחינוך','השרה למדע','השרה לחדשנות','השרה לשוויון חברתי',
    'שרת הקליטה','שרת התחבורה','שרת הבריאות','שרת החינוך','שרת התיירות',
    'שרת התרבות והספורט','סגן שר','סגנית שר','יושב ראש הכנסת','יו״ר הכנסת',
    'יו"ר הכנסת','סגן יושב ראש הכנסת','מנהל הוועדה','ראש הרשות','מפקד מחוז',
    'ממונה על','ראש הממשלה','עו"ד','שרת המשפטים','שר המודיעין',
}

# HEBREW NUMBER DICTIONARIES
HEBREW_UNITS = {
    "אפס": 0, "אחד": 1, "אחת": 1, "שניים": 2, "שתיים": 2, 
    "שלוש": 3, "שלושה": 3, "ארבע": 4, "ארבעה": 4,
    "חמש": 5, "חמישה": 5, "שש": 6, "שישה": 6, "שבע": 7,
    "שבעה": 7, "שמונה": 8, "תשע": 9, "תשעה": 9
}

HEBREW_TENS = {
    "עשר": 10, "עשרים": 20, "שלושים": 30, "ארבעים": 40,
    "חמישים": 50, "שישים": 60, "שבעים": 70,
    "שמונים": 80, "תשעים": 90
}

HEBROW_HUNDREDS = {
    "מאה": 100, "מאתיים": 200, "שלוש מאות": 300, "ארבע מאות": 400,
    "חמש מאות": 500, "שש מאות": 600, "שבע מאות": 700,
    "שמונה מאות": 800, "תשע מאות": 900,
}

# ────────────────────────────────────────────────
#      FILE LOADER
# ────────────────────────────────────────────────

class FileLoader:
    def __init__(self, path:str = PATH):
        self.path = path
    
    def ListFiles(self):
        return os.listdir(self.path)


# ────────────────────────────────────────────────
#      FILENAME PARSER
# ────────────────────────────────────────────────

class FileParser:
    def __init__(self, filename:str):
        self.filename = filename    

    def parse_filename(self):
        match = re.match(r'^(\d{1,2})_pt([mv])_.*\.(docx|txt)$', self.filename)
        if not match:
            return None

        knesset_number = int(match.group(1))
        protocol_type  = "plenary" if match.group(2) == "m" else "committee"

        return knesset_number, protocol_type


# ────────────────────────────────────────────────
#      PROTOCOL PARSER (Supports DOCX + TXT)
# ────────────────────────────────────────────────

class Protocol:
    def __init__(self, knesset_number, protocol_type, filename):
        self.knesset_number = knesset_number
        self.protocol_type = protocol_type
        self.filepath = os.path.join(PATH, filename)
        self.extension = filename.split(".")[-1].lower()

        self.text_lines = self._load_text()

        self.protocol_number = self._extract_protocol_number()
        self.yor_hankest = self._extract_yor()
        self.colon_sentences = self._extract_colon_sentences()

    # ---------------- LOAD FILE -----------------
    def _load_text(self):
        if self.extension == "docx":
            try:
                doc = Document(self.filepath)
                return [p.text for p in doc.paragraphs]
            except:
                return []
        elif self.extension == "txt":
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return f.read().splitlines()
            except:
                return []
        return []

    def _get_full_text(self):
        return "\n".join(self.text_lines)

    # ---------------- PROTOCOL NUMBER -----------------
    def _extract_protocol_number(self):
        text = self._get_full_text()

        m1 = re.search(r"פרוטוקול\s+מס'?[\s:]*([0-9]+)", text)
        if m1:
            return int(m1.group(1))

        m2 = re.search(r"הישיבה\s+([א-ת\s\-־]+)", text)
        if m2:
            return self._hebrew_number_to_int(m2.group(1))

        return None

    # ---------------- HEBREW NUMBER -----------------
    def _hebrew_number_to_int(self, text):
        words = text.replace("־", " ").replace("-", " ").split()
        total = 0
        for w in words:
            if w in HEBROW_HUNDREDS: total += HEBROW_HUNDREDS[w]
            elif w in HEBREW_TENS: total += HEBREW_TENS[w]
            elif w in HEBREW_UNITS: total += HEBREW_UNITS[w]
        return total if total > 0 else None

    # ---------------- YOR / SPEAKER -----------------
    def _extract_yor(self):
        text = self._get_full_text()
        match = re.search(r'היו"ר\s+([א-ת\"\'״׳\.\-\s]+?)(:|\n|$)', text)
        if match:
            return match.group(1).strip()
        return "Unknown"

    # ---------------- SPEAKERS -----------------
    def _extract_colon_sentences(self):
        results = []

        for line in self.text_lines:
            text = line.strip()

            text = re.sub(r"<<.*?>>", "", text)
            text = re.sub(r"<.*?>", "", text)

        
            if ':' not in text:
                continue

            # Take everything before the first colon
            clean = text.split(":", 1)[0].strip()

            if clean in INVALID_TALKERS_NAMES:
                continue

            for suffix in GET_RID_OF_SUFFIX:
                clean = clean.replace(suffix, "").strip()

            # Remove parentheses content
            clean = re.sub(r'\([^)]*\)', '', clean).strip()
            clean = " ".join(clean.split())

            parts = clean.split()
            if 2 <= len(parts) <= 5:
                results.append(clean)

        # UNIQUE + LENGTH FILTER
        final = []
        seen = set()
        for s in results:
            if s not in seen and 5 < len(s) < 50:
                seen.add(s)
                final.append(s)

        # Debug empty files
        if not final:
            with open("debug_files.txt", "a", encoding="utf-8") as f:
                f.write(self.filepath + "\n")

        return final


def process_file(filename):
    parsed = FileParser(filename).parse_filename()
    if not parsed:
        return None

    knesset_number, protocol_type = parsed

    protocol = Protocol(knesset_number, protocol_type, filename)

    return {
        "knesset_number": protocol.knesset_number,
        "protocol_type": protocol.protocol_type,
        "protocol_number": protocol.protocol_number,
        "chair": protocol.yor_hankest,
        "filename": filename,
        "Speakers": protocol.colon_sentences
    }



if __name__ == "__main__":
    fl = FileLoader()
    files = fl.ListFiles()

    
    with Pool(processes=cpu_count()) as pool:
        protocols = pool.map(process_file, files)

    # protocols = [process_file(f) for f in files if f.endswith((".docx", ".txt"))]
    # protocols = [p for p in protocols if p is not None]

    for p in protocols:
        print(f"Knesset: {p['knesset_number']}, Type: {p['protocol_type']}, "
              f"Protocol Number: {p['protocol_number']}, Chair: {p['chair']}, "
              f"File: {p['filename']}, Speakers: {p['Speakers']}")
