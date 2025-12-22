from ast import Dict
from collections import Counter
from typing import Tuple, Set
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.metrics import f1_score
import sys
import os
import string
import json
import logging
import random
import numpy as np
random.seed(42)
np.random.seed(42)


UNIGRAMS = (1,1)
BIGRAMS = (1,2)
TRIGRAMS = (1,7)
MAX_FEATURES = 2000
MINDEF = 3
MAXDEF = 0.9
NGRAMS =TRIGRAMS


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("output.log", mode='w', encoding='utf-8')
    ]
)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%H:%M:%S',
#     handlers=[
#         logging.FileHandler("output.log", mode='w', encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

logger = logging.getLogger(__name__)




def tune_logistic_regression(X, y, C_values=None, max_iter_values=None, cv=5, metric="f1"):
    """
    Performs grid search for LogisticRegression over C and max_iter.
    
    Args:
        X: Feature matrix
        y: Labels
        C_values: List of C values to try
        max_iter_values: List of max_iter values to try
        cv: Number of cross-validation folds
        metric: Scoring metric (default "f1", uses macro average)
    
    Returns:
        best_lr: LogisticRegression instance with best parameters
        best_params: dict of best parameters
        best_score: best F1 score
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10]
    if max_iter_values is None:
        max_iter_values = [500, 1000, 2000]

    best_score = 0
    best_params = {}
    best_lr = None

    for C in C_values:
        for max_iter in max_iter_values:
            lr = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            y_pred = cross_val_predict(lr, X, y, cv=cv, n_jobs=1)
            score = f1_score(y, y_pred, average="macro")
            logger.info(f"LR: C={C}, max_iter={max_iter} -> F1={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'max_iter': max_iter}
                best_lr = lr

    logger.info(f"\nBest LR params: {best_params}, Best F1={best_score:.4f}")
    
    return best_lr, best_params, best_score



def char_level_features(text):
    if not text:
        return [0, 0, 0, 0]

    total = len(text)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    punct = sum(c in string.punctuation for c in text)
    avg_word_len = np.mean([len(w) for w in text.split()]) if text.split() else 0

    return [
        letters / total,
        digits / total,
        punct / total,
        avg_word_len
    ]


def lexical_features(tokens):
    if not tokens:
        return [0, 0, 0, 0]

    total = len(tokens)
    unique = len(set(tokens))
    hapax = sum(1 for t in set(tokens) if tokens.count(t) == 1)

    long_words = sum(len(t) > 6 for t in tokens)

    return [
        unique / total,          
        hapax / total,          
        unique,                 
        long_words / total       
    ]



HEB_FUNCTION_WORDS = {
    "של", "על", "עם", "אל", "אם", "כי", "גם", "לא",
    "כן", "זה", "הוא", "היא", "אנחנו", "אני", "אתם"
}

def function_word_features(tokens):
    if not tokens:
        return [0, 0]

    fw = [t for t in tokens if t in HEB_FUNCTION_WORDS]

    return [
        len(fw) / len(tokens),
        len(set(fw)) / max(len(fw), 1)
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
    words = text.split()
    wc = len(words)

    commas = text.count(',')
    parentheses = text.count('(') + text.count(')')

    return [
        commas / max(wc, 1),
        parentheses / max(wc, 1),
        max(len(w) for w in words) if words else 0
    ]



EMPHASIS_WORDS = {"מאוד", "באמת", "בהחלט", "חייבים"}


def emphasis_features(text):
    return [
        text.count('!') / max(len(text), 1),
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
    text = record.get("sentence_text", "")
    tokens = text.split()

    features = []

    features += lexical_features(tokens)
    features += function_word_features(tokens)
    features += discourse_features(text)
    features += structural_features(text)
    features += emphasis_features(text)
    features += numeric_features(text)
    features += char_level_features(text)
    features += role_features(record)

    features += [
        len(tokens),
        len(text),
        int('?' in text)
    ]

    return features



def train_and_evaluate(X, y, task_name, feature_name):
    logger.info(f"\n===== {task_name} | {feature_name} =====")
    
    k_vals = [1, 3, 5, 7, 9, 11, 15, 21]
    best_k = find_best_k(X, y, k_vals)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X, y) 
    
    y_pred_knn = cross_val_predict(knn, X, y, cv=5)
    logger.info(f"--- KNN (k={best_k}) ---\n{classification_report(y, y_pred_knn, digits=3)}")

    
    lr_best, best_params, _ = tune_logistic_regression(X, y)
    lr_best.fit(X, y)
    
    y_pred_lr = cross_val_predict(lr_best, X, y, cv=5)
    logger.info(f"--- Logistic Regression {best_params} ---\n{classification_report(y, y_pred_lr, digits=3)}")
    
    return {"lr": lr_best, "knn": knn}




def scale_custom_features(X_custom):
    scaler = StandardScaler()
    return scaler.fit_transform(X_custom)


def combine_features(X_text, X_custom):
    return hstack([X_text, X_custom])




def find_best_k(X, y, k_values, metric="f1"):
    scores = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(
            knn, X, y, cv=5, n_jobs=1
        )
        score = f1_score(y, y_pred, average="macro")
        scores[k] = score
        logger.info(f"k={k}: {metric}={score:.4f}")

    best_k = max(scores, key=scores.get)
    logger.info(f"Best k = {best_k} (score={scores[best_k]:.4f})")

    return best_k


def run_all_experiments(task, task_name) -> Dict:
    """Runs all feature combinations and returns a dictionary of all fitted models."""
    models_registry = {}
    y = np.array(task.labels)

    # models_registry["BOW"] = train_and_evaluate(task.X_bow, y, task_name, "BOW")
    # models_registry["TF-IDF"] = train_and_evaluate(task.X_tfidf, y, task_name, "TF-IDF")
    
  
    X_custom_scaled = scale_custom_features(task.X_custom)
    # models_registry["Custom"] = train_and_evaluate(X_custom_scaled, y, task_name, "Custom")
    
    # X_bow_custom = combine_features(task.X_bow, X_custom_scaled)
    # models_registry["BOW_Custom"] = train_and_evaluate(X_bow_custom, y, task_name, "BOW + Custom")

    X_tfidf_custom = combine_features(task.X_tfidf, X_custom_scaled)
    models_registry["TFIDF_Custom"] = train_and_evaluate(X_tfidf_custom, y, task_name, "TF-IDF + Custom")

    return models_registry


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

   
        logger.info(f"Testing MINDEF: {MINDEF}, MAXDEF: {MAXDEF}, Ngram: {NGRAMS}, Max Features: {MAX_FEATURES}")
        text_data = [r['sentence_text'] for r in self.records]
        self.vectorizer_bow = CountVectorizer(min_df=MINDEF, max_df=MAXDEF, ngram_range=NGRAMS, max_features=MAX_FEATURES)
        self.X_bow = self.vectorizer_bow.fit_transform(text_data)
        logger.info(f"BOW Vector shape: {self.X_bow.shape}")

        self.vectorizer_tfidf = TfidfVectorizer(min_df=MINDEF, max_df=MAXDEF, ngram_range=NGRAMS, max_features=MAX_FEATURES)
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
        logger.info(f"Testing MINDEF: {MINDEF}, MAXDEF: {MAXDEF}, Ngram: {NGRAMS}, Max Features: {MAX_FEATURES}")
        self.vectorizer_bow = CountVectorizer(min_df=MINDEF, max_df=MAXDEF, ngram_range=NGRAMS, max_features=MAX_FEATURES)
        self.X_bow = self.vectorizer_bow.fit_transform(text_data)
        logger.info(f"BOW Vector shape: {self.X_bow.shape}")

        self.vectorizer_tfidf = TfidfVectorizer(min_df=MINDEF, max_df=MAXDEF, ngram_range=NGRAMS, max_features=MAX_FEATURES)
        self.X_tfidf = self.vectorizer_tfidf.fit_transform(text_data)
        logger.info(f"TF-IDF Vector shape: {self.X_tfidf.shape}")
        
        custom_features = [extract_custom_features_from_record(r) for r in self.records]
        self.X_custom = np.array(custom_features)
        logger.info(f"Custom Vector shape: {self.X_custom.shape}")
    


    def train_final_model(self):
        """
        Trains a final model on the entire balanced dataset using TF-IDF only.
        NOTE: We use TF-IDF because the stage 5 input file has NO metadata (no roles, no speakers),
        so we cannot use the custom features that rely on those fields.
        """
        logger.info("--- Training Final Model for Inference (TF-IDF + LR) ---")
        
        text_data = [r['sentence_text'] for r in self.records]
        y = self.labels
        
        self.final_vectorizer = TfidfVectorizer(min_df=5)
        X = self.final_vectorizer.fit_transform(text_data)
        
        self.final_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.final_model.fit(X, y)
        logger.info("Final model trained successfully.")


    def predict_all_and_save(self, models_dict, input_file_path, output_dir):
        """
        Runs inference using ALL trained models and feature strategies.
        Saves one file per model and a single classification_results.txt with majority voting.
        """
        logger.info(f"Inference started on: {input_file_path}")

        with open(input_file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        os.makedirs(output_dir, exist_ok=True)
        label_map = {0: 'first', 1: 'second', 2: 'other'}

        all_model_preds = []  # 

        for feature_name, models in models_dict.items():
            logger.info(f"--- Using features: {feature_name} ---")


            if feature_name == "BOW":
                X_test = self.vectorizer_bow.transform(sentences)
            elif feature_name == "TF-IDF":
                X_test = self.vectorizer_tfidf.transform(sentences)
            elif feature_name == "Custom":
                X_custom = scale_custom_features(np.array([
                    extract_custom_features_from_record({"sentence_text": s})
                    for s in sentences
                ]))
                X_test = X_custom
            elif feature_name == "BOW_Custom":
                X_bow = self.vectorizer_bow.transform(sentences)
                X_custom = scale_custom_features(np.array([
                    extract_custom_features_from_record({"sentence_text": s})
                    for s in sentences
                ]))
                X_test = hstack([X_bow, X_custom])
            elif feature_name == "TFIDF_Custom":
                X_tfidf = self.vectorizer_tfidf.transform(sentences)
                X_custom = scale_custom_features(np.array([
                    extract_custom_features_from_record({"sentence_text": s})
                    for s in sentences
                ]))
                X_test = hstack([X_tfidf, X_custom])
            else:
                logger.warning(f"Unknown feature type: {feature_name}")
                continue


            for model_name, model in models.items():
                logger.info(f"Predicting with {feature_name} + {model_name}")
                preds = model.predict(X_test)
                all_model_preds.append(preds)  

                out_file = f"{feature_name}_{model_name}_predictions.txt"
                out_path = os.path.join(output_dir, out_file)
                # with open(out_path, 'w', encoding='utf-8') as f:
                #     for p in preds:
                #         f.write(f"{label_map[p]}\n")
                # logger.info(f"Saved predictions to {out_path}")


        logger.info("Computing majority vote for classification_results.txt")
        all_model_preds_array = np.array(all_model_preds)  
        majority_preds = []

        for i in range(len(sentences)):
            votes = all_model_preds_array[:, i]
            most_common_label = Counter(votes).most_common(1)[0][0]
            majority_preds.append(most_common_label)

        majority_file = os.path.join(output_dir, 'classification_results.txt')
        with open(majority_file, 'w', encoding='utf-8') as f:
            for p in majority_preds:
                f.write(f"{label_map[p]}\n")

        logger.info(f"Saved majority vote predictions to {majority_file}")



class TaskHandler:
    """Main handler to run the classification tasks."""
    def __init__(self, corpus_file: str):
        self.corpus_file = corpus_file
        self.binary_task = None
        self.multi_task = None

    def run(self, sentences_path=None, output_dir=None):
        top_speakers = TopTwoSpeakers(self.corpus_file)
    
        if top_speakers.speaker1 and top_speakers.speaker2:
            ln1 = top_speakers.speaker1.split()[-1]
            ln2 = top_speakers.speaker2.split()[-1]
            
            aliases1, aliases2 = get_variations(self.corpus_file, ln1, ln2)
            
            self.binary_task = BinaryClassificationTask(aliases1, aliases2)
            self.binary_task.load_data(self.corpus_file)
            self.binary_task.balance_classes()
            self.binary_task.create_features()
            
            self.multi_task = MultiClassClassificationTask(aliases1, aliases2)
            self.multi_task.load_data(self.corpus_file)
            self.multi_task.balance_classes()
            self.multi_task.create_features()

            logger.info("\n==============================")
            logger.info(" RUNNING BINARY CLASSIFICATION ")
            logger.info("==============================")

            self.binary_models = run_all_experiments(self.binary_task, "Binary Task")


            logger.info("\n==============================")
            logger.info(" RUNNING MULTI-CLASS CLASSIFICATION ")
            logger.info("==============================")

            all_multi_models = run_all_experiments(self.multi_task, "Multi-Class")

            if sentences_path and output_dir:
                logger.info("\n=== STAGE 5: INFERENCE (Reusing Experiment Models) ===")
                self.multi_task.predict_all_and_save(all_multi_models, sentences_path, output_dir)


if __name__ == "__main__":


    if len(sys.argv) < 4:
        logger.error("Error: Missing arguments.")
        logger.error("Usage: python knesset_speaker_classification.py <corpus_path> <sentences_path> <output_dir>")
        sys.exit(1)
    

    corpus_path = sys.argv[1]
    sentences_path = sys.argv[2]
    output_dir = sys.argv[3]


    if os.path.exists(corpus_path):
        handler = TaskHandler(corpus_file=corpus_path)
        handler.run(sentences_path=sentences_path, output_dir=output_dir)
    else:
        logger.error(f"Corpus file not found at: {corpus_path}")
        sys.exit(1)