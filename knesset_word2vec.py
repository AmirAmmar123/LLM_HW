import json
import os
import re
import argparse
import logging
from typing import List, Iterator
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s : %(levelname)s : %(message)s',
#     handlers=[
#         logging.FileHandler("temp.log", encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )


logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)
logger.propagate = False



SELECTED_INDICES = [230, 1118, 6605, 8111, 3954, 4049, 52, 1066, 162, 1628]

WORDS = ["יום", "אישה", "דרך", "ארוך", "תוכנית", "אוהב", "אסור", "איתן", "זכות"] 

REPLACEMENT_TASKS = [
            {
                "id": 1,
                "orig": "בשעות הקרובות נפתח את הדיון בסעיף הבא שעל סדר היום",
                "targets": [
                    {"word": "בשעות", "pos": ["בשניות","בדקות","בימים","בשנים"], "neg": ["נסגרים"]},
                    {"word": "בסעיף", "pos": ["בנושא", "בחלק", ], "neg": []}
                ]
            },
            {
                "id": 2,
                "orig": "אבקש מהנוכחים להתיישב כדי שנוכל להתחיל",
                "targets": [
                    {"word": "אבקש", "pos": ["קורא"], "neg": ["פונה"]},
                    {"word": "להתיישב", "pos": ["לשבת", "שקט"], "neg": ["להשתתף"]}
                ]
            },
            {
                "id": 3,
                "orig": "אנו מודים לצוות המקצועי על עבודתו",
                "targets": [
                    {"word": "לצוות", "pos": ["למערך","לעובדים", "לאנשים","לחייל",], "neg": []},
                    {"word": "עבודתו", "pos": ["תרומתו", "עזרתו", "גבר"], "neg": ["שמו"]}
                ]
            },
            {
                "id": 4,
                "orig": "הנושא יועבר להמשך טיפול בוועדת המשנה",
                "targets": [
                    {"word": "הנושא", "pos": ["החוק", "העניין"], "neg": []},
                    {"word": "טיפול", "pos": [], "neg": []}
                ]
            },
            {
                "id": 5,
                "orig": "ההצעה הובאה להצבעה ואושרה",
                "targets": [
                    {"word": "ההצעה", "pos": ["הבקשה", "הדרישה"], "neg": ["הגישה"]},
                    {"word": "הובאה", "pos": ["נשלחה"], "neg": []},
                    {"word": "ואושרה", "pos": [], "neg": ["נדחתה"]}
                ]
            }
        ]


class KnessetCorpusIterator:
    """
    Provides a memory-efficient iterator to stream sentences from the JSONL corpus.
    Ensures the entire dataset is not loaded into memory at once.
    """
    def __init__(self, file_path: str):
        """
        Initializes the iterator with the corpus file path and a regex to keep Hebrew characters only.
        """
        self.file_path = file_path
        self.hebrew_regex = re.compile(r'[^א-ת\s]')

    def __iter__(self) -> Iterator[List[str]]:
        """
        Iterates over the file line by line, cleans non-Hebrew tokens, and yields tokenized lists.
        Removes punctuation and numbers as required by the assignment instructions.
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
            logging.error(f"Error during corpus iteration: {e}")
            raise

    def get_raw_sentences(self, limit: int = 10000) -> List[dict]:
        """
        Retrieves original sentences alongside their tokenized versions for similarity tasks.
        Includes a filter to ensure sentences have at least 4 valid tokens.
        """
        results = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    raw_text = data.get('sentence_text', data.get('text', ''))
                    clean_text = self.hebrew_regex.sub('', raw_text)
                    tokens = clean_text.split()
                    if len(tokens) >= 4:
                        results.append({'raw': raw_text, 'tokens': tokens})
                    if len(results) >= limit:
                        break
        except Exception as e:
            logging.error(f"Error retrieving raw sentences: {e}")
        return results

class Word2VecManager:
    """
    Manages the Word2Vec model lifecycle, including training, persistence, and similarity analysis.
    """
    def __init__(self, vector_size: int = 50, window: int = 5, min_count: int = 1):
        """
        Initializes model parameters such as vector size (50) and window size (5).
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def train_model(self, sentences: Iterator[List[str]]):
        """
        Trains the Word2Vec model using the provided sentence iterator.
        """
        logging.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )

    def save(self, output_dir: str, filename: str = "knesset_word2vec.model"):
        """
        Saves the trained model to the specified output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, filename)
        self.model.save(path)
        logging.info(f"Model saved to {path}")

    def load_model(self, model_path: str):
        """
        Loads an existing Word2Vec model from the disk.
        """
        logging.info(f"Loading existing model from {model_path}...")
        self.model = Word2Vec.load(model_path)

    def run_word_similarity(self, output_dir: str):
        """
        Identifies top 5 similar words using concise list comprehensions.
        Calculates manual similarity scores as required in Part B, Section A .
        """
        vocab = self.model.wv.index_to_key
        
        try:
            with open(os.path.join(output_dir, "knesset_similar_words.txt"), 'w', encoding='utf-8') as f:
                for t in [w for w in WORDS if w in self.model.wv]:
                    top = sorted([(v, self.model.wv.similarity(t, v)) for v in vocab if v != t], key=lambda x: x[1], reverse=True)[:5]
                    f.write(f"{t}: {', '.join([f'({w}, {s:.4f})' for w, s in top])}\n")
        except Exception as e:
            logging.error(f"Error in concise similarity task: {e}")


    def run_sentence_similarity(self, corpus_samples: List[dict], output_dir: str):
        """
        Computes sentence embeddings by averaging word vectors.
        Finds the most similar sentence for 10 selected samples using cosine similarity.
        """
        def get_avg_embedding(tokens):
            vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
            return np.mean(vectors, axis=0) if vectors else None

        valid_data = []
        for sample in corpus_samples:
            emb = get_avg_embedding(sample['tokens'])
            if emb is not None:
                valid_data.append({'raw': sample['raw'], 'emb': emb})

        if len(valid_data) < 10: return
        
        embeddings_matrix = np.array([item['emb'] for item in valid_data])
        output_path = os.path.join(output_dir, "knesset_similar_sentences.txt")
        

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx in SELECTED_INDICES:
                if idx >= len(valid_data):
                    continue  # safety check

                target_emb = valid_data[idx]['emb'].reshape(1, -1)
                similarities = cosine_similarity(target_emb, embeddings_matrix)[0]

                similarities[idx] = -1

                best_idx = np.argmax(similarities)

                f.write(
                    f"{valid_data[idx]['raw']}: most similar sentence: {valid_data[best_idx]['raw']}\n"
                )

    def run_red_words(self, output_dir: str):
        """
            Executes the semantic substitution task for Part B, Section D.
            This implementation adheres to the specific constraints:
            1. Selecting from the top 3 similar candidates.
            2. Utilizing anchor words for better context or OOV handling.
            3. Maintaining the exact output format specified in the instructions.
        """

        
        output_path = os.path.join(output_dir, "red_words_sentences.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in REPLACEMENT_TASKS:
                    current_sentence = item["orig"]
                    applied_replacements = []
                    
                    for target_config in item["targets"]:
                        original_word = target_config["word"]
                        
          
                        pos_anchors = target_config["pos"]
                        if original_word in self.model.wv:
                            pos_anchors.append(original_word)
                        
                
                        candidates = self.model.wv.most_similar(
                            positive=pos_anchors, 
                            negative=target_config["neg"], 
                            topn=3
                        )
                        
           
                        chosen_substitute = original_word
                        for candidate, score in candidates:
                            if candidate != original_word:
                                chosen_substitute = candidate
                                break
      
                        current_sentence = current_sentence.replace(original_word, chosen_substitute)
                        applied_replacements.append(f"({original_word}: {chosen_substitute})")
                    

                    f.write(f"{item['id']}: {item['orig']}: {current_sentence}\n")
                    f.write(f"replaced words: {', '.join(applied_replacements)}\n")
                    
        except Exception as e:
            logging.error(f"Semantic substitution module encountered an error: {e}")

def main():
    """
    Main entry point for the script. Checks for existing model or initiates training.
    """
    parser = argparse.ArgumentParser(description="Knesset Corpus Word2Vec Analysis")
    parser.add_argument("corpus_path", help="Path to input .jsonl file")
    parser.add_argument("output_dir", help="Directory for output files")
    args = parser.parse_args()

    model_filename = "knesset_word2vec.model"
    model_path = os.path.join(args.output_dir, model_filename)

    try:
        manager = Word2VecManager(vector_size=50, window=5, min_count=1)

        if os.path.exists(model_path):
            manager.load_model(model_path)
        else:
            corpus_iterator = KnessetCorpusIterator(args.corpus_path)
            manager.train_model(corpus_iterator)
            manager.save(args.output_dir, model_filename)

        
        logging.info("Executing Part B analysis tasks...")
        manager.run_word_similarity(args.output_dir)
        
        corpus_iterator = KnessetCorpusIterator(args.corpus_path)
        raw_samples = corpus_iterator.get_raw_sentences(limit=120000)
        manager.run_sentence_similarity(raw_samples, args.output_dir)
        
        manager.run_red_words(args.output_dir)
        logging.info("Analysis completed successfully.")

    except Exception as e:
        logging.critical(f"Process failed: {e}")

if __name__ == "__main__":
    main()