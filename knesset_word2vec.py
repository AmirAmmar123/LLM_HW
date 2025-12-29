import json
import os
import re
import argparse
import logging
from typing import List, Iterator
from gensim.models import Word2Vec


DEBUG = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s',
    handlers=[
        logging.FileHandler("temp.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class KnessetCorpusIterator:
    """
    Memory-efficient iterator that streams sentences from the JSONL corpus file.
    This approach prevents loading the entire dataset into RAM, which is crucial for large scale NLP tasks.
    """
    def __init__(self, file_path: str):
        """
        Initializes the iterator with the path to the Knesset corpus.
        """
        self.file_path = file_path
        self.hebrew_regex = re.compile(r'[^א-ת\s]')

    def __iter__(self) -> Iterator[List[str]]:
        """
        Iterates over the file line by line, cleans the text, and yields tokenized sentences.
        Remove non-word tokens like punctuation and numbers...etc.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)

                    raw_text = data.get('sentence_text', data.get('text', ''))
                    

                    clean_text = self.hebrew_regex.sub('', raw_text)
                    tokens = clean_text.split()
                    
                    if tokens:
                        yield tokens
        except Exception as e:
            logging.error(f"Failed to process corpus file: {e}")
            raise

class Word2VecManager:
    """
    Handles the initialization, training, and persistence of the Word2Vec model.
    """
    def __init__(self, vector_size: int = 50, window: int = 5, min_count: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def train_model(self, sentences: Iterator[List[str]]):
        """
        Trains the Word2Vec model using the provided sentence iterator.
        Uses multi-threading (workers) to optimize performance.
        """
        logging.info("Starting Word2Vec training process...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        logging.info("Training finished successfully.")

    def save(self, output_dir: str, filename: str = "knesset_word2vec.model"):
        """
        Persists the trained model to the specified directory.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before saving.")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        path = os.path.join(output_dir, filename)
        self.model.save(path)
        logging.info(f"Model saved at: {path}")

def main():

    parser = argparse.ArgumentParser(description="Knesset Corpus Word2Vec Trainer")
    parser.add_argument("corpus_path", help="Path to the input .jsonl corpus file")
    parser.add_argument("output_dir", help="Directory where the model and outputs will be saved")
    
    args = parser.parse_args()

    try:

        corpus_iterator = KnessetCorpusIterator(args.corpus_path)
        

        manager = Word2VecManager(vector_size=50, window=5, min_count=1)
        

        manager.train_model(corpus_iterator)
        manager.save(args.output_dir)

        word_vectors = manager.model.wv
        logging.info(f"Total vocabulary size: {len(word_vectors)}")

        total_trained_tokens = sum(manager.model.wv.get_vecattr(word, "count")
                           for word in manager.model.wv.key_to_index)

        logging.info(f"Total trained tokens (after min_count): {total_trained_tokens}")
        
        if DEBUG:
            print(word_vectors['ישראל'])
        
    except Exception as e:
        logging.critical(f"Execution failed: {e}")

if __name__ == "__main__":
    main()