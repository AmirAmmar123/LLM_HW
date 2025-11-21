import sys
from processing_kenest_corpus import process_file, FileLoader

# Redirect stdout to file
sys.stdout = open('output_analysis.txt', 'w', encoding='utf-8')

if __name__ == "__main__":
    fl = FileLoader()
    files = fl.ListFiles()
    
    # Process only a subset of files for faster analysis if needed, 
    # but let's run all as per request or a representative sample
    # Taking first 20 files for quick check or all if fast enough.
    # Let's do all to see robust results.
    
    protocols = [process_file(f) for f in files]
    protocols = [p for p in protocols if p is not None]

    for p in protocols:
        if not p:
            continue

        print(f"Knesset: {p['knesset_number']}, Type: {p['protocol_type']}, Protocol Number: {p['protocol_number']}, Chair: {p['chair']}, file: {p['filename']}")
        print(f"Speakers: {p['Speakers']}")
        speeches = p.get("Speeches", [])
        print(f"Total speeches: {len(speeches)}")
        for speech in speeches[:3]:
            snippet = speech['text'][:200].replace('\n', ' ')
            print(f"- {speech['speaker']}: {snippet}...")
        print()

