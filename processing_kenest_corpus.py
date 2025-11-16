import re
import os
from docx import Document

PATH = "./knesset_protocols"

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
    


    def _extract_protocol_number(self):
        """Extract the protocol number from the document."""
        try:
            self.doc = Document(self.filepath)
        except Exception as e:
            print(f"Error opening {self.filepath}: {e}")
            return -1

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

  
        

class ProtocolsCollection:
    def __init__(self):
        self.protocols = []

    def add_protocol(self, protocol):
        self.protocols.append(protocol)


if __name__ == "__main__":
    fl = FileLoader()
    pc = ProtocolsCollection()
    for filename in fl.ListFiles():
        result = FileParser(filename).parse_filename()
        if result:
            knesset_number, protocol_type = result
            pc.add_protocol(Protocol(knesset_number, protocol_type, filename))
        else:
            print(f"Filename: {filename} => Invalid format")
    
    for protocol in pc.protocols:
        print(f"Knesset: {protocol.knesset_number}, Type: {protocol.protocol_type}, Protocol Number: {protocol.protocol_number}","File name:", protocol.filepath, "Chair Name:", protocol.chair_name)
