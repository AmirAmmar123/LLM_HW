import sys
import os
import torch
from transformers import pipeline, utils

utils.logging.set_verbosity_error()

def run_dictabert(input_path, output_dir):
    device = 0 if torch.cuda.is_available() else -1
    
    mask_filler = pipeline(
        "fill-mask", 
        model="dicta-il/dictabert", 
        device=device
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, "dictabert_results.txt")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                original_masked_sent = line.strip()
                if not original_masked_sent:
                    continue
                
                ready_for_bert = original_masked_sent.replace("[*]", "[MASK]")
                
                results = mask_filler(ready_for_bert)
                
                if isinstance(results[0], dict):
                    results = [results]
                
                predicted_tokens = []
                final_sentence = ready_for_bert
                
                for res in results:
                    top_prediction = res[0]['token_str']
                    predicted_tokens.append(top_prediction)
                    final_sentence = final_sentence.replace("[MASK]", top_prediction, 1)

                f_out.write(f"masked_sentence: {original_masked_sent}\n")
                f_out.write(f"dictaBERT_sentence: {final_sentence}\n")
                f_out.write(f"dictaBERT tokens: {','.join(predicted_tokens)}\n")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python knesset_dictabert.py <path/to/masked_sampled_sents.txt> <path/to/output_dir>")
    else:
        run_dictabert(sys.argv[1], sys.argv[2])