import json
import logging
import random
import numpy as np
from collections import Counter
from typing import Tuple, Set
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


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



def lexical_features(tokens):
    """"Extracts lexical diversity features from tokens."""
    if not tokens:
        return [0, 0, 0]

    unique = len(set(tokens))
    total = len(tokens)

    hapax = sum(1 for t in set(tokens) if tokens.count(t) == 1)

    return [
        unique / total,         
        hapax / total,           
        unique                  
    ]



HEB_FUNCTION_WORDS = {
    "של", "על", "עם", "אל", "אם", "כי", "גם", "לא",
    "כן", "זה", "הוא", "היא", "אנחנו", "אני", "אתם"
}

def function_word_features(tokens):
    """Extracts function word features from tokens."""
    if not tokens:
        return [0, 0]

    fw_count = sum(1 for t in tokens if t in HEB_FUNCTION_WORDS)

    return [
        fw_count / len(tokens),
        fw_count
    ]



DISCOURSE_MARKERS = {
    "על כן", "לכן", "ברשותכם", "אדוני היושב ראש",
    "אני מבקש", "אני רוצה", "נדמה לי", "כמובן"
}

def discourse_features(text):
    """Extracts discourse marker features from text."""
    return [
        int(marker in text) for marker in DISCOURSE_MARKERS
    ]




def structural_features(text):
    """Extracts structural features from text."""
    words = text.split()

    return [
        text.count(','),                
        text.count('–'),                
        text.count('(') + text.count(')'),
        sum(len(w) > 6 for w in words),   
        max(len(w) for w in words) if words else 0
    ]



EMPHASIS_WORDS = {"מאוד", "באמת", "בהחלט", "חייבים"}

def emphasis_features(text):
    """Extracts emphasis features from text."""
    return [
        int('!' in text),
        sum(text.count(w) for w in EMPHASIS_WORDS)
    ]



def numeric_features(text):
    """Extracts numeric features from text."""
    digits = sum(c.isdigit() for c in text)

    return [
        digits,
        digits / max(len(text), 1),
        int("אלף" in text or "מיליון" in text)
    ]



def role_features(record):
    """Extracts role-based features from the record."""
    speaker = record.get("speaker_name", "")
    chair = record.get("protocol_chairman", "")

    return [
        int(speaker == chair),
        int("ראש" in speaker),
        int("שר" in speaker),
        int("חבר הכנסת" in speaker)
    ]




def extract_custom_features_from_record(record: dict) -> list:
    """Extracts custom features from a record."""
    text = record.get("sentence_text", "")
    tokens = text.split()

    features = []

    features += lexical_features(tokens)
    features += function_word_features(tokens)
    features += discourse_features(text)
    features += structural_features(text)
    features += emphasis_features(text)
    features += numeric_features(text)
    features += role_features(record)

    features += [
        len(tokens),                           
        len(text),                              
        len(set(tokens)) / max(len(tokens), 1), 
        int('?' in text)                       
    ]

    return features


def train_and_evaluate(X, y, task_name, feature_name):
    logger.info(f"\n===== {task_name} | {feature_name} =====")

    knn = KNeighborsClassifier(n_neighbors=5)

    y_pred_knn = cross_val_predict(
        knn, X, y, cv=5, n_jobs=-1
    )

    logger.info("\n--- KNN ---")
    logger.info(classification_report(y, y_pred_knn, digits=3))


    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )


    y_pred_lr = cross_val_predict(
        lr, X, y, cv=5, n_jobs=-1
    )

    logger.info("\n--- Logistic Regression ---")
    logger.info(classification_report(y, y_pred_lr, digits=3))



def scale_custom_features(X_custom):
    scaler = StandardScaler()
    return scaler.fit_transform(X_custom)


def combine_features(X_text, X_custom):
    return hstack([X_text, X_custom])




def run_all_experiments(task, task_name):
    y = np.array(task.labels)

    train_and_evaluate(
        task.X_bow,
        y,
        task_name,
        "BOW"
    )

    train_and_evaluate(
        task.X_tfidf,
        y,
        task_name,
        "TF-IDF"
    )

    X_custom_scaled = scale_custom_features(task.X_custom)
    train_and_evaluate(
        X_custom_scaled,
        y,
        task_name,
        "Custom Features"
    )

    X_bow_custom = combine_features(
        task.X_bow,
        X_custom_scaled
    )
    train_and_evaluate(
        X_bow_custom,
        y,
        task_name,
        "BOW + Custom"
    )

    X_tfidf_custom = combine_features(
        task.X_tfidf,
        X_custom_scaled
    )
    train_and_evaluate(
        X_tfidf_custom,
        y,
        task_name,
        "TF-IDF + Custom"
    )



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

            logger.info("\n==============================")
            logger.info(" RUNNING BINARY CLASSIFICATION ")
            logger.info("==============================")

            run_all_experiments(
                self.binray_task,
                task_name="Binary Classification"
            )

            logger.info("\n==============================")
            logger.info(" RUNNING MULTI-CLASS CLASSIFICATION ")
            logger.info("==============================")

            run_all_experiments(
                self.multi_task,
                task_name="Multi-Class Classification"
            )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    corpus_file = 'knesset_corpus.jsonl' 

    handler = TaskHandler(corpus_file)
    handler.run()