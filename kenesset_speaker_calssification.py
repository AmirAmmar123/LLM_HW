import json
import logging
import random
import numpy as np
from collections import Counter
from typing import Tuple, Set
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("output.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_custom_features_from_record(record: dict) -> list:
    """
    Helper function to extract stylistic and metadata features from a single record.
    Returns a list of numerical features.
    """
    text = record.get('sentence_text', '')
    
    word_count = len(text.split())
    
    char_count = len(text.replace(" ", ""))
    avg_word_len = char_count / word_count if word_count > 0 else 0
    
    num_commas = text.count(',')
    num_questions = text.count('?')
    num_exclamations = text.count('!')
    num_quotes = text.count('"')
    
    is_plenary = 1 if record.get('protocol_type') == 'plenary' else 0
    
    try:
        knesset_num = int(record.get('knesset_number', 0))
    except (ValueError, TypeError):
        knesset_num = 0
        
    speaker = record.get('speaker_name', '').strip()
    chairman = record.get('protocol_chairman', '').strip()
    is_chairman_speaking = 1 if (speaker and chairman and speaker == chairman) else 0

    return [
        word_count, 
        avg_word_len, 
        num_commas, 
        num_questions, 
        num_exclamations, 
        num_quotes,
        is_plenary, 
        knesset_num,
        is_chairman_speaking
    ]

class TopTwoSpeakers:
    """Finds the two most common speakers in the corpus."""
    def __init__(self, corpus_path: str):
        self.speaker1, self.speaker2 = self._get_top_two_speakers(corpus_path)

    def _get_top_two_speakers(self, corpus_path) -> Tuple[str, str]:
        speaker_counter = Counter()
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        speaker = record.get('speaker_name', '').strip()
                        if speaker:
                            speaker_counter[speaker] += 1
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.error(f"The file '{corpus_path}' was not found.")
            return None, None

        most_common = speaker_counter.most_common(20)
        if not most_common:
            logger.warning("No speakers found in the file.")
            return None, None

        logger.info("--- Top 20 Speakers ---")
        for i, (name, count) in enumerate(most_common, 1):
            logger.info(f"{i}. {name}: {count}")
    
        return most_common[0][0], most_common[1][0]

def get_variations(corpus_path: str, last_name_1: str, last_name_2: str) -> Tuple[Set[str], Set[str]]:
    """
    Finds all variations of two given last names in the corpus.
    """
    variations_1 = set()
    variations_2 = set()
    logger.info(f"Searching for variations of '{last_name_1}' and '{last_name_2}'")
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    speaker = record.get('speaker_name', '').strip()
                    if not speaker: continue 
                    if last_name_1 in speaker: variations_1.add(speaker)
                    if last_name_2 in speaker: variations_2.add(speaker)
                except json.JSONDecodeError: continue
        logger.info(f"Found {len(variations_1)} variations for {last_name_1}: {variations_1}")
        logger.info(f"Found {len(variations_2)} variations for {last_name_2}: {variations_2}")
        return variations_1, variations_2
    except FileNotFoundError:
        logger.error(f"File {corpus_path} not found.")
        return set(), set()

class BinaryClassificationTask:
    """Handles loading, balancing, and feature extraction for binary classification."""
    def __init__(self, speaker1: set, speaker2: set):
        self.speaker1_aliases = speaker1
        self.speaker2_aliases = speaker2
        self.records = [] 
        self.labels = [] 
        
        self.X_bow = None
        self.X_tfidf = None
        self.X_custom = None
        self.vectorizer_bow = None
        self.vectorizer_tfidf = None

    def load_data(self, corpus_path):
        count_s1 = 0
        count_s2 = 0
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        speaker = record.get('speaker_name', '').strip()
                        text = record.get('sentence_text', '').strip()
                        if not speaker or not text: continue

                        if speaker in self.speaker1_aliases:
                            self.records.append(record) 
                            self.labels.append(0) 
                            count_s1 += 1
                        elif speaker in self.speaker2_aliases:
                            self.records.append(record) 
                            self.labels.append(1) 
                            count_s2 += 1
                    except json.JSONDecodeError: continue
            logger.info("--- Binary Data Loaded ---")
            logger.info(f"Class 1 (Label 0): {count_s1}")
            logger.info(f"Class 2 (Label 1): {count_s2}")
            logger.info(f"Total: {len(self.records)}")
        except FileNotFoundError:
            logger.error(f"File {corpus_path} not found.")

    def balance_classes(self):
        logger.info("--- Balancing Binary Classes ---")
        data = list(zip(self.records, self.labels))
        class0 = [d for d in data if d[1] == 0]
        class1 = [d for d in data if d[1] == 1]
        
        len0, len1 = len(class0), len(class1)
        logger.info(f"Counts before: Class 0={len0}, Class 1={len1}")
        
        min_len = min(len0, len1)
        class0_down = random.sample(class0, min_len)
        class1_down = random.sample(class1, min_len)
        
        balanced_data = class0_down + class1_down
        random.shuffle(balanced_data)
        
        self.records, self.labels = zip(*balanced_data)
        self.records = list(self.records)
        self.labels = list(self.labels)
        
        logger.info(f"Counts after: Class 0={self.labels.count(0)}, Class 1={self.labels.count(1)}")

    def create_features(self):
        logger.info("--- Creating Feature Vectors (Binary) ---")
        
        text_data = [r['sentence_text'] for r in self.records]
        
        self.vectorizer_bow = CountVectorizer(min_df=5)
        self.X_bow = self.vectorizer_bow.fit_transform(text_data)
        logger.info(f"BOW Vector shape: {self.X_bow.shape}")

        self.vectorizer_tfidf = TfidfVectorizer(min_df=5)
        self.X_tfidf = self.vectorizer_tfidf.fit_transform(text_data)
        logger.info(f"TF-IDF Vector shape: {self.X_tfidf.shape}")
        
        custom_features = [extract_custom_features_from_record(r) for r in self.records]
        self.X_custom = np.array(custom_features)
        logger.info(f"Custom Vector shape: {self.X_custom.shape}")

class MultiClassClassificationTask:
    """Handles loading, balancing, and feature extraction for multi-class classification."""
    def __init__(self, speaker1_aliases, speaker2_aliases):
        self.speaker1_aliases = speaker1_aliases
        self.speaker2_aliases = speaker2_aliases
        self.records = [] 
        self.labels = []

        self.X_bow = None
        self.X_tfidf = None
        self.X_custom = None
        self.vectorizer_bow = None
        self.vectorizer_tfidf = None

    def load_data(self, corpus_path):
        logger.info("Loading Multi-Class Data...")
        count_s1, count_s2, count_other = 0, 0, 0
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        speaker = record.get('speaker_name', '').strip()
                        text = record.get('sentence_text', '').strip()
                        if not speaker or not text: continue

                        if speaker in self.speaker1_aliases:
                            self.records.append(record)
                            self.labels.append(0) 
                            count_s1 += 1
                        elif speaker in self.speaker2_aliases:
                            self.records.append(record)
                            self.labels.append(1) 
                            count_s2 += 1
                        else:
                            self.records.append(record) 
                            self.labels.append(2)
                            count_other += 1
                    except json.JSONDecodeError: continue
            logger.info("--- Multi-Class Data Loaded ---")
            logger.info(f"Class 0: {count_s1}, Class 1: {count_s2}, Class 2: {count_other}")
            logger.info(f"Total: {len(self.records)}")
        except FileNotFoundError:
            logger.error(f"File {corpus_path} not found.")

    def balance_classes(self):
        logger.info("--- Balancing Multi-Class Classes ---")
        data = list(zip(self.records, self.labels))
        class0 = [d for d in data if d[1] == 0]
        class1 = [d for d in data if d[1] == 1]
        class2 = [d for d in data if d[1] == 2] 
        
        lengths = [len(class0), len(class1), len(class2)]
        logger.info(f"Counts before: 0={lengths[0]}, 1={lengths[1]}, 2={lengths[2]}")
        
        min_len = min(lengths)
        logger.info(f"Down-sampling to {min_len} per class.")
        
        c0_down = random.sample(class0, min_len)
        c1_down = random.sample(class1, min_len)
        c2_down = random.sample(class2, min_len)
        
        balanced_data = c0_down + c1_down + c2_down
        random.shuffle(balanced_data)
        
        self.records, self.labels = zip(*balanced_data)
        self.records = list(self.records)
        self.labels = list(self.labels)
        
        logger.info(f"Counts after: 0={self.labels.count(0)}, 1={self.labels.count(1)}, 2={self.labels.count(2)}") 

    def create_features(self):
        logger.info("--- Creating Feature Vectors (Multi-Class) ---")
        
        text_data = [r['sentence_text'] for r in self.records]
        
        self.vectorizer_bow = CountVectorizer(min_df=5)
        self.X_bow = self.vectorizer_bow.fit_transform(text_data)
        logger.info(f"BOW Vector shape: {self.X_bow.shape}")

        self.vectorizer_tfidf = TfidfVectorizer(min_df=5)
        self.X_tfidf = self.vectorizer_tfidf.fit_transform(text_data)
        logger.info(f"TF-IDF Vector shape: {self.X_tfidf.shape}")
        
        custom_features = [extract_custom_features_from_record(r) for r in self.records]
        self.X_custom = np.array(custom_features)
        logger.info(f"Custom Vector shape: {self.X_custom.shape}")

class TaskHandler:
    """Main handler to run the classification tasks."""
    def __init__(self, corpus_file: str):
        self.corpus_file = corpus_file
        self.binray_task = None
        self.multi_task = None

    def run(self):
        top_speakers = TopTwoSpeakers(self.corpus_file)
    
        if top_speakers.speaker1 and top_speakers.speaker2:
            ln1 = top_speakers.speaker1.split()[-1]
            ln2 = top_speakers.speaker2.split()[-1]
            
            aliases1, aliases2 = get_variations(self.corpus_file, ln1, ln2)
            
            self.binray_task = BinaryClassificationTask(aliases1, aliases2)
            self.binray_task.load_data(self.corpus_file)
            self.binray_task.balance_classes()
            self.binray_task.create_features()
            
            self.multi_task = MultiClassClassificationTask(aliases1, aliases2)
            self.multi_task.load_data(self.corpus_file)
            self.multi_task.balance_classes()
            self.multi_task.create_features()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    corpus_file = 'knesset_corpus.jsonl' 

    handler = TaskHandler(corpus_file)
    handler.run()