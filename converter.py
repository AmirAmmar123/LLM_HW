import os
from docx import Document

PATH = r"./knesset_protocols"
TXT_FOLDER = os.path.join(r'./', "to_txt")

# Create the output folder if it doesn't exist
os.makedirs(TXT_FOLDER, exist_ok=True)

def docx_to_txt(docx_path, txt_path):
    doc = Document(docx_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                f.write(text + "\n")

if __name__ == "__main__":
    files = os.listdir(PATH)
    for filename in files:
        if filename.endswith(".docx"):
            docx_path = os.path.join(PATH, filename)
            txt_path = os.path.join(TXT_FOLDER, os.path.splitext(filename)[0] + ".txt")
            try:
                docx_to_txt(docx_path, txt_path)
            except Exception as e:
                print(f"Error converting {filename} to TXT: {e}")
