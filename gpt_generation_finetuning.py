import sys
import os
import torch
import random
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk


def tokenize_reviews(dataset, tokenizer, max_length=150): 
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
 
    def tokenize_function(examples): 
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length) 
 
    tokenized_dataset = dataset.map(tokenize_function, batched=True) 
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  
    return tokenized_dataset 


FLAG = False 
SAMPLE = 100 #TODO: change back to 100
def main():


    if len(sys.argv) != 4:
        print("Usage: python gpt_generation_finetuning.py <imdb_subset_path> <output_file_path> <saved_models_dir>")
        sys.exit(1)

    subset_path = sys.argv[1]
    output_file_path = sys.argv[2]
    saved_models_dir = sys.argv[3]

    if FLAG:
        print(f"Loading dataset from: {subset_path}")
    
    try:
        dataset = load_from_disk(subset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {subset_path}")
        return

    if FLAG:
        print("Splitting dataset into positive and negative subsets...")
    
    positive_dataset = dataset.filter(lambda x: x['label'] == 1).shuffle(seed=42).select(range(SAMPLE))
    negative_dataset = dataset.filter(lambda x: x['label'] == 0).shuffle(seed=42).select(range(SAMPLE))

    if FLAG:
        print(f"Positive samples: {len(positive_dataset)}")
        print(f"Negative samples: {len(negative_dataset)}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def train_gpt2_model(train_dataset, output_dir_name):
        final_save_path = os.path.join(saved_models_dir, output_dir_name)

        if FLAG and os.path.exists(final_save_path) and os.path.exists(os.path.join(final_save_path, "config.json")):
            print(f"DEBUG: Found existing model at '{final_save_path}'. Skipping training.")
            return final_save_path

        if FLAG:
            print(f"Starting training for {output_dir_name}...")
        
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        tokenized_dataset = tokenize_reviews(train_dataset, tokenizer)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=os.path.join(saved_models_dir, output_dir_name, "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=3,             
            per_device_train_batch_size=4,  
            save_steps=500,
            save_total_limit=1,
            prediction_loss_only=True,
            use_cpu=not torch.cuda.is_available(), 
            logging_steps=10
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        
        trainer.model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        if FLAG:
            print(f"Model saved to {final_save_path}")
        return final_save_path

    pos_model_path = train_gpt2_model(positive_dataset, "positive_model")
    neg_model_path = train_gpt2_model(negative_dataset, "negative_model")

    prompt = "The movie was"
    
    def generate_reviews(model_path, num_reviews=10):
        if FLAG:
            print(f"Generating reviews from model at {model_path}...")

        model = GPT2LMHeadModel.from_pretrained(model_path)
        local_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        if local_tokenizer.pad_token is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token

        input_ids = local_tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = input_ids.ne(local_tokenizer.pad_token_id).long() 

        reviews = []
        for _ in range(num_reviews):
 
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,           
                    do_sample=True,           
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    repetition_penalty=1.2, 
                    pad_token_id=local_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated_text = local_tokenizer.decode(output[0], skip_special_tokens=True)
            clean_text = generated_text.replace("\n", " ")
            reviews.append(clean_text)
        
        return reviews

    pos_reviews = generate_reviews(pos_model_path)
    neg_reviews = generate_reviews(neg_model_path)

    if FLAG:
        print(f"Writing results to {output_file_path}...")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("Reviews generated by positive model:\n")
        for i, review in enumerate(pos_reviews, 1):
            f.write(f"{i}. {review}\n")
        
        f.write("Reviews generated by negative model:\n")
        for i, review in enumerate(neg_reviews, 1):
            f.write(f"{i}. {review}\n")

    if FLAG:
        print("Done!")

if __name__ == "__main__":
    main()