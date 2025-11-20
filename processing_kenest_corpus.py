import re
import os
from docx import Document
from multiprocessing import Pool, cpu_count
import json



PATH = r"./knesset_protocols"


INVALID_TALKERS_NAMES = {
    "חברי הוועדה",
    "מוזמנים",
    "קריאה",
    "אני מבקש לשאול"
}

GET_RID_OF_SUFFIX ={
    'היו"ר',
    'היו”ר',
}


HEBREW_UNITS = {
    "אפס": 0,
    "אחד": 1, "אחת": 1,
    "שניים": 2, "שתיים": 2, "שני": 2, "שתי": 2,
    "שלוש": 3, "שלושה": 3,
    "ארבע": 4, "ארבעה": 4,
    "חמש": 5, "חמישה": 5,
    "שש": 6, "שישה": 6,
    "שבע": 7, "שבעה": 7,
    "שמונה": 8,
    "תשע": 9, "תשעה": 9
}

HEBREW_TENS = {
    "עשר": 10, "עשרה": 10,
    "עשרים": 20,
    "שלושים": 30,
    "ארבעים": 40,
    "חמישים": 50,
    "שישים": 60,
    "שבעים": 70,
    "שמונים": 80,
    "תשעים": 90,
}

HEBROW_HUNDREDS = {
    "מאה": 100,
    "מאתיים": 200,
    "שלוש מאות": 300,
    "שלוש־מאות": 300,   
    "ארבע מאות": 400,
    "חמש מאות": 500,
    "שש מאות": 600,
    "שבע מאות": 700,
    "שמונה מאות": 800,
    "תשע מאות": 900
}


def process_file(filename):
    """Process a single file and return a dictionary with protocol info or None."""
    result = FileParser(filename).parse_filename()
    if result:
        knesset_number, protocol_type = result
        protocol = Protocol(knesset_number, protocol_type, filename)
        if protocol.protocol_number is not None:
            return {
                "knesset_number": protocol.knesset_number,
                "protocol_type": protocol.protocol_type,
                "protocol_number": protocol.protocol_number,
                "chair": protocol.yor_hankest,
                "filename": filename,
                "Speakers": protocol.colon_sentences
            }
    else:
        print(f"Filename: {filename} => Invalid format")
    return None



class FileLoader:
    """Class to load files from a specified directory."""
    def __init__(self, path:str = PATH):
        self.path = path
    
    def ListFiles(self):
        return os.listdir(self.path)


class FileParser:
    """Class to parse filenames and extract metadata."""

    def __init__(self, filename:str):
        self.filename = filename    

        
    def parse_filename(self):
        if not self.filename:
            return None
        match = re.match(r'^(\d{1,2})_pt([mv])_.*\.docx$', self.filename)
        if not match:
            return None

        knesset_number = int(match.group(1))
        letter = match.group(2)
        protocol_type = "plenary" if letter == 'm' else "committee"

        return knesset_number, protocol_type


class Protocol:
    def __init__(self, knesset_number: int, protocol_type: str, filename: str):
        self.knesset_number = knesset_number
        self.protocol_type = protocol_type
        self.filepath = os.path.join(PATH, filename)
        self.protocol_number = self._extract_protocol_number()
        if self.protocol_number is None:
            return
        
        self.yor_hankest = self._extract_yor()
        self.colon_sentences = self._extract_colon_sentences()

    def _extract_protocol_number(self):
        """Extract the protocol number from the document."""
        try:
            self.doc = Document(self.filepath)
        except Exception as e:
            print(f"Error opening {self.filepath}: {e}")
            return None

        text = "\n".join(p.text for p in self.doc.paragraphs)

 
        m1 = re.search(r"פרוטוקול\s+מס'?[\s:]*([0-9]+)", text)
    
        m2 = re.search(r"הישיבה\s+([א-ת\s\-־]+)", text)

        if m1:
            return int(m1.group(1))
        if m2:
            return self._hebrew_number_to_int(m2.group(1))

        return -1
    

    def _hebrew_number_to_int(self, text):
        """Convert Hebrew number words into integers"""

        text = text.replace("־", " ").replace("-", " ").replace("–", " ")
        words = text.split()


        if "של" in words:
            words = words[:words.index("של")]

   
        if text.strip() == "אלף":
            return 1000

        total = 0
        i = 0
        while i < len(words):
            w = words[i]

        
            while w.startswith(("ה", "ו")) and len(w) > 1:
                w = w[1:]

  
            if w == "ו":
                i += 1
                continue

      
            if i + 1 < len(words):
                next_w = words[i+1].lstrip("ו").lstrip("ה")
                pair = f"{w} {next_w}"
                if pair in HEBROW_HUNDREDS:
                    total += HEBROW_HUNDREDS[pair]
                    i += 2
                    continue


            if w in HEBROW_HUNDREDS:
                total += HEBROW_HUNDREDS[w]
            elif w in HEBREW_TENS:
                total += HEBREW_TENS[w]
            elif w in HEBREW_UNITS:
                total += HEBREW_UNITS[w]

            i += 1

        return total if total > 0 else -1

    
    def _extract_yor(self):
        """Extract only the first chairperson's full name from the document."""
        text = "\n".join(p.text for p in self.doc.paragraphs)


        match = re.search(r'היו"ר\s+([א-ת\"\'״׳\.\-\s]+?)(:|\n|מוזמנים|$)', text)
        if match:
            full_name = match.group(1).strip()

            for line in full_name.splitlines():
                line = line.strip()
                if line:
                    return line
        return "Unknown"


    def _extract_colon_sentences(self):
        """
        Extract paragraphs that:
        - end with a colon ':'
        - contain at least one run that is bold OR underlined
        """
        try:
            doc = self.doc if hasattr(self, "doc") else Document(self.filepath)
        except Exception as e:
            print(f"Error opening {self.filepath} for colon sentences: {e}")
            return []

        results = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text.endswith(':'):
                continue

            clean_text = text.rstrip(':').strip()
            if clean_text not in INVALID_TALKERS_NAMES:
                for suffix in GET_RID_OF_SUFFIX:
                    clean_text = clean_text.replace(suffix, "").strip()
                    
                clean_text = re.sub(r'\([^)]*\)', '', clean_text).strip()
                clean_text = " ".join(clean_text.split())
                results.append(clean_text)

        seen = set()
        unique = []
        for s in results:
            if len(s) < 50 and len(s) > 5 and  s not in seen :
                seen.add(s)
                unique.append(s)

        return unique


        

class ProtocolsCollection:
    def __init__(self):
        self.protocols = []

    def add_protocol(self, protocol):
        self.protocols.append(protocol)




if __name__ == "__main__":
    fl = FileLoader()
    files = fl.ListFiles()


    # with Pool(processes=cpu_count()) as pool:
    #     protocols = pool.map(process_file, files)

    protocols = [process_file(f) for f in files]
    protocols = [p for p in protocols if p is not None]


    for p in protocols:
        if p:
            print(f"Knesset: {p['knesset_number']}, Type: {p['protocol_type']}, Protocol Number: {p['protocol_number']}, Chair: {p['chair']}, file: {p['filename']}, Speakers: {p['Speakers']}")
