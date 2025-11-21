import sys
from processing_kenest_corpus import process_file

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "15_ptv_490845.docx"

    result = process_file(filename)
    if not result:
        print(f"Could not process file: {filename}")
        return

    speeches = result.get("Speeches", [])
    output_path = "sentence_test_output.txt"

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"File: {filename}\n")
        out.write(f"Knesset: {result['knesset_number']}, Type: {result['protocol_type']}, Protocol Number: {result['protocol_number']}\n")
        out.write(f"Total speeches: {len(speeches)}\n\n")

        for speech in speeches:
            out.write(f"Speaker: {speech['speaker']}\n")
            sentences = speech.get("sentences", [])
            if not sentences:
                out.write("  [no sentences detected]\n\n")
                continue
            for idx, sentence in enumerate(sentences, start=1):
                out.write(f"  [{idx}] {sentence}\n")
            out.write("\n")

    print(f"Wrote sentence breakdown to {output_path}")

if __name__ == "__main__":
    main()

