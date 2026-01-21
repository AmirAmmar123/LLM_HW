import sys
import os
import random
import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def normalize_prediction(pred):
    pred = pred.strip().lower()
    if pred == "positive":
        return "positive"
    if pred == "negative":
        return "negative"
    return "invalid"

def test_prompts(review_text, fs_pos_text, fs_neg_text, fs_pos_example2, fs_neg_example2):
    prompts = {
                    "zero-shot": (
                        "Classify the sentiment of the following movie review "
                        "as positive or negative:\n\n"
                        f"{review_text}"
                    ),
                    "few-shot": (
                        f"Review: {fs_pos_text}\n"
                        "Sentiment: positive\n\n"
                        f"Review: {fs_neg_text}\n"
                        "Sentiment: negative\n\n"
                        f"Review: {review_text}\n"
                        "Sentiment:"
                    ),
                    "instruction-based": (
                        "You are a sentiment classifier.\n"
                        "Determine whether the following movie review is positive or negative.\n"
                        "Answer with exactly one word: positive or negative.\n"
                        f"Review: {review_text}"
                    )
        }
    
    prompts1 = {
                    "zero-shot": (
                        "Classify the sentiment of the following movie review "
                        "as positive or negative:\n\n"
                        f"{review_text}"
                    ),
                    "few-shot": (
                        f"Review: {fs_pos_text}\n"
                        "return: positive\n\n"
                        f"Review: {fs_neg_text}\n"
                        "return: negative\n\n"
                        f"Review: {review_text}\n"
                        "return:"
                    ),
                    "instruction-based": (
                        "You are a sentiment classifier.\n"
                        "Determine whether the following movie review is positive or negative.\n"
                        "Answer with exactly one word: positive or negative.\n"
                        f"Review: {review_text}"
                    )
        }
    
    prompts2 = {
                    "zero-shot": (
                        "Classify the sentiment of the following movie review "
                        "as positive or negative:\n\n"
                        f"{review_text}"
                    ),
                        "few-shot": (
                        f"Review: {fs_pos_text}\n"
                        "return: positive\n\n"
                        f"Review: {fs_neg_text}\n"
                        "return: negative\n\n"
                        "Review: Absolutely fantastic! The acting and story were top-notch.\n"
                        "return: positive\n\n"
                        "Review: Terrible plot, awful acting, completely disappointing.\n"
                        "return: negative\n\n"
                        f"Review: {review_text}\n"
                        "return:"
                    ),
                    "instruction-based": (
                        "You are a sentiment classifier.\n"
                        "Determine whether the following movie review is positive or negative.\n"
                        "Answer with exactly one word: positive or negative.\n"
                        f"Review: {review_text}"
                    )
        }
    
    
    
    prompts3 = {
                    "zero-shot": (
                        "Binary Classify the sentiment positive or negative"
                        f"{review_text}"
                    ),
                    "few-shot": (
                        f"Review: {fs_pos_text}\n"
                        "return: positive\n\n"
                        f"Review: {fs_neg_text}\n"
                        "return: negative\n\n"
                        f"Review: {review_text}\n"
                        "return:"
                    ),
                    "instruction-based": (
                        "You are a sentiment classifier.\n"
                        "Determine whether the following movie review is positive or negative.\n"
                        "Answer with exactly one word: positive or negative.\n"
                        f"Review: {review_text}"
                    )

            }
    
    prompt4 = {
            "zero-shot": (
                "Classify the sentiment of the following movie review as either positive or negative.\n\n"
                f"{review_text}\n"
                "Answer with exactly one word: positive or negative."
            ),

            "few-shot": (
                "Review: Absolutely fantastic! The acting and story were top-notch.\n"
                "Return: positive\n\n"
                "Review: Terrible plot, awful acting, completely disappointing.\n"
                "Return: negative\n\n"
                "Review: I laughed a lot, but some scenes were boring.\n"
                "Return: positive\n\n"
                "Review: Bad acting and story ruined it.\n"
                "Return: negative\n\n"
                f"Review: {review_text}\n"
                "Return:"
            ),

            "instruction-based": (
                "You are a sentiment classifier. Determine whether the following movie review is positive or negative.\n"
                "Answer with exactly one word: positive or negative.\n"
                f"Review: {review_text}"
            ),

        }
    

    prompts5 = {
                    "zero-shot": (
                        "Classify the sentiment of the following movie review "
                        "as positive or negative:\n\n"
                        f"{review_text}"
                    ),
                    "few-shot": (
                        f"Review: {fs_pos_text}\n"
                        "return: positive\n\n"
                        f"Review: {fs_neg_text}\n"
                        "return: negative\n\n"
                        f"Review: {fs_pos_example2}\n"
                        "return: positive\n\n"
                        f"Review: {fs_neg_example2}\n"
                        "return: negative\n\n"
                        f"Review: {review_text}\n"
                        "return:"
                    ),
                    "instruction-based": (
                        "You are a sentiment classifier.\n"
                        "Determine whether the following movie review is positive or negative.\n"
                        "Answer with exactly one word: positive or negative.\n"
                        f"Review: {review_text}"
                    )
        }

    return prompts, prompts1, prompts2, prompts3, prompt4, prompts5

def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python flan_t5_prompt_engineering.py "
            "<path/to/imdb_subset> <path/to/flan_t5_imdb_results.txt>"
        )
        sys.exit(1)

    subset_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        if os.path.exists(subset_path):
            dataset = load_from_disk(subset_path)
        else:
            dataset = load_dataset("imdb")["train"].shuffle(seed=42).select(range(500))
            dataset.save_to_disk(subset_path)

        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()

        pos_reviews = dataset.filter(lambda x: x["label"] == 1)
        neg_reviews = dataset.filter(lambda x: x["label"] == 0)

        sampled_pos = pos_reviews.shuffle(seed=42).select(range(25))
        sampled_neg = neg_reviews.shuffle(seed=42).select(range(25))

        # fs_pos_text = "This movie was amazing with great acting and a compelling story."
        # fs_neg_text = "This movie was boring, poorly written, and a complete waste of time."

        fs_pos_text = (
                        "This movie was absolutely amazing from start to finish. "
                        "The acting was convincing and emotionally powerful, the story was engaging, "
                        "and the characters were well developed. I found myself completely immersed "
                        "in the plot and would definitely recommend this film to anyone who enjoys "
                        "thoughtful and well-made movies."
                    )

        fs_neg_text = (
                        "This movie was extremely boring and poorly executed. "
                        "The story lacked direction, the acting felt flat and unconvincing, "
                        "and many scenes dragged on without purpose. I struggled to stay interested "
                        "and overall felt that watching this film was a complete waste of time."
                    )

        
        fs_pos_example2 = pos_reviews.shuffle(seed=123).select(range(1))[0]["text"]
        fs_neg_example2 = neg_reviews.shuffle(seed=123).select(range(1))[0]["text"]

        fs_pos_example3 = pos_reviews.shuffle(seed=123).select(range(1))[0]["text"]
        fs_neg_example3 = neg_reviews.shuffle(seed=123).select(range(1))[0]["text"]

        test_samples = []
        for i in range(25):
            test_samples.append((sampled_pos[i], "positive"))
            test_samples.append((sampled_neg[i], "negative"))

        random.seed(42)
        random.shuffle(test_samples)

        correct = {
            "zero-shot": 0,
            "few-shot": 0,
            "instruction-based": 0
        }
        total = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, (sample, true_label) in enumerate(test_samples, 1):
                review_text = sample["text"]


                prompts = test_prompts(review_text, fs_pos_text, fs_neg_text, fs_pos_example2, fs_neg_example2)[5]

                results = {}

                for p_type, p_text in prompts.items():
                    inputs = tokenizer(
                        p_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096
                    )

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=10,
                        )

                    raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    prediction = normalize_prediction(raw_pred)
                    results[p_type] = prediction

                    if prediction == true_label:
                        correct[p_type] += 1

                total += 1

                f.write(f"Review {idx}: {review_text}\n")
                f.write(f"Review {idx} true label: {true_label}\n")
                f.write(f"Review {idx} zero-shot: {results['zero-shot']}\n")
                f.write(f"Review {idx} few-shot: {results['few-shot']}\n")
                f.write(
                    f"Review {idx} instruction-based: "
                    f"{results['instruction-based']}\n"
                )

            print("=== Accuracy Results ===\n")
            for p_type in correct:
                acc = correct[p_type] / total
                print(f"{p_type} accuracy: {acc:.2f}")


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
