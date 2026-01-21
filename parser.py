import re
from collections import defaultdict

def parse_sentiment_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    reviews = {}
    blocks = re.split(r"(?=Review \d+:)", text)

    for block in blocks:
        if not block.strip():
            continue

        review_id_match = re.search(r"Review (\d+):", block)
        if not review_id_match:
            continue
        review_id = int(review_id_match.group(1))

        data = {}
        true_label = re.search(r"true label:\s*(\w+)", block)
        zero = re.search(r"zero-shot:\s*(\w+)", block)
        few = re.search(r"few-shot:\s*(\w+)", block)
        instr = re.search(r"instruction-based:\s*(\w+)", block)

        if true_label:
            data["true"] = true_label.group(1).lower()
        if zero:
            data["zero-shot"] = zero.group(1).lower()
        if few:
            data["few-shot"] = few.group(1).lower()
        if instr:
            data["instruction-based"] = instr.group(1).lower()

        reviews[review_id] = data

    return reviews

# ----------------- Create binary table -----------------
def create_binary_table_and_acc(reviews):
    table = []
    # For accuracy calculation
    model_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for review_id in sorted(reviews.keys()):
        data = reviews[review_id]
        true = data.get("true")
        row = {"Review": review_id}

        for model in ["zero-shot", "few-shot", "instruction-based"]:
            pred = data.get(model)
            correct = int(pred == true)
            row[model] = correct

            # Update accuracy counts
            if pred is not None:
                model_scores[model]["total"] += 1
                if correct:
                    model_scores[model]["correct"] += 1

        table.append(row)

    # Compute accuracy percentages
    acc = {}
    for model, stats in model_scores.items():
        acc[model] = stats["correct"] / stats["total"] if stats["total"] else 0

    return table, acc

# ----------------- Print functions -----------------
def print_binary_table(table):
    print(f"{'Review':6s} {'Zero-Shot':10s} {'Few-Shot':10s} {'Instruction':12s}")
    print("-" * 40)
    for row in table:
        print(f"{row['Review']:6d} {row['zero-shot']:10d} {row['few-shot']:10d} {row['instruction-based']:12d}")

def print_accuracy(acc):
    print("\nModel Accuracy")
    print("-" * 25)
    for model, a in acc.items():
        print(f"{model:20s}: {a:.2%}")

# ----------------- Main -----------------
if __name__ == "__main__":
    path = "/home/amir/LLM/HW5/flan_t5_imdb_results_3.txt"
    reviews = parse_sentiment_file(path)

    table, acc = create_binary_table_and_acc(reviews)

    print_binary_table(table)
    print_accuracy(acc)
