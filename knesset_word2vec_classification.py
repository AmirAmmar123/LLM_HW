import json
import sys
import logging
import random
import numpy as np
from collections import Counter
from typing import Tuple, Set

from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score

random.seed(42)
np.random.seed(42)


K_VALUES = [1,2,3,4,5,6,7,8, 9,10,11,12,13,14, 15,16,17,18,19,20, 21]
# logging.getLogger(__name__).addHandler(logging.NullHandler())
# logger = logging.getLogger(__name__)
# logger.propagate = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("temp2.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)



def find_best_k(X, y, k_values, metric="f1_macro"):
    """Finds the best k for KNN using cross-validation."""
    if X is None or len(X) == 0:
        raise ValueError("Empty feature matrix X")

    if len(set(y)) < 2:
        raise ValueError("Need at least two classes for classification")

    scores = {}

    for k in k_values:
        try:
            if k >= len(y):
                logger.warning(f"Skipping k={k}: k must be < number of samples")
                continue

            knn = KNeighborsClassifier(n_neighbors=k)
            y_pred = cross_val_predict(knn, X, y, cv=5, n_jobs=1)

            score = f1_score(y, y_pred, average="macro")
            scores[k] = score
            logger.info(f"k={k}: {metric}={score:.4f}")

        except Exception as e:
            logger.error(f"Failed evaluating k={k}: {e}")

    if not scores:
        raise RuntimeError("Failed to evaluate any k value")

    best_k = max(scores, key=scores.get)
    logger.info(f"Best k = {best_k} (score={scores[best_k]:.4f})")

    return best_k

def sentence_embedding(sentence: str, word_vectors) -> np.ndarray:
    """Computes the sentence embedding by averaging word vectors."""
    if not sentence:
        return np.zeros(word_vectors.vector_size)

    tokens = sentence.split()
    vectors = [word_vectors[t] for t in tokens if t in word_vectors]

    if not vectors:
        logger.debug("Sentence has no known tokens – returning zero vector")
        return np.zeros(word_vectors.vector_size)

    return np.mean(vectors, axis=0)


class TopTwoSpeakers:
    """Finds the two most frequent speakers in the corpus."""
    def __init__(self, corpus_path: str):
        self.speaker1, self.speaker2 = self._get_top_two(corpus_path)

    def _get_top_two(self, corpus_path) -> Tuple[str, str]:
        counter = Counter()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    speaker = record.get("speaker_name", "").strip()
                    if speaker:
                        counter[speaker] += 1
                except json.JSONDecodeError:
                    continue

        most_common = counter.most_common(2)
        if len(most_common) < 2:
            raise RuntimeError("Not enough speakers in corpus")

        logger.info(f"Top speakers: {most_common[0][0]}, {most_common[1][0]}")
        return most_common[0][0], most_common[1][0]


def get_variations(corpus_path: str, last_name_1: str, last_name_2: str) -> Tuple[Set[str], Set[str]]:
    """Finds all name variations containing the given last names."""
    v1, v2 = set(), set()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                speaker = record.get("speaker_name", "").strip()
                if last_name_1 in speaker:
                    v1.add(speaker)
                if last_name_2 in speaker:
                    v2.add(speaker)
            except json.JSONDecodeError:
                continue

    logger.info(f"Speaker 1 aliases: {v1}")
    logger.info(f"Speaker 2 aliases: {v2}")
    return v1, v2


class BinarySentenceClassificationTask:
    """Binary classification task for sentences by two speakers."""
    def __init__(self, speaker1_aliases: Set[str], speaker2_aliases: Set[str]):
        self.speaker1_aliases = speaker1_aliases
        self.speaker2_aliases = speaker2_aliases
        self.sentences = []
        self.labels = []
        self.X = None

    def load_data(self, corpus_path: str):
        c0 = c1 = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    speaker = record.get("speaker_name", "").strip()
                    text = record.get("sentence_text", "").strip()
                    if not speaker or not text:
                        continue

                    if speaker in self.speaker1_aliases:
                        self.sentences.append(text)
                        self.labels.append(0)
                        c0 += 1
                    elif speaker in self.speaker2_aliases:
                        self.sentences.append(text)
                        self.labels.append(1)
                        c1 += 1
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded sentences: class0={c0}, class1={c1}")

    def balance(self):
        """Balances the dataset to have equal number of samples per class."""
        data = list(zip(self.sentences, self.labels))
        c0 = [d for d in data if d[1] == 0]
        c1 = [d for d in data if d[1] == 1]

        min_len = min(len(c0), len(c1))
        balanced = random.sample(c0, min_len) + random.sample(c1, min_len)
        random.shuffle(balanced)

        self.sentences, self.labels = zip(*balanced)
        self.sentences = list(self.sentences)
        self.labels = list(self.labels)

        logger.info(f"Balanced dataset size: {len(self.labels)}")

    def create_sentence_embeddings(self, word_vectors):
        X = [sentence_embedding(s, word_vectors) for s in self.sentences]
        self.X = np.vstack(X)
        logger.info(f"Sentence embedding matrix shape: {self.X.shape}")


def run_knn(task):
    """Runs KNN classification on the given task."""
    X = task.X
    y = np.array(task.labels)

  
    best_k = find_best_k(X, y, K_VALUES)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    y_pred = cross_val_predict(knn, X, y, cv=5)

    logger.info("\n===== Binary Classification | Sentence Embeddings (Word2Vec) =====")
    logger.info(f"Best k chosen by CV: {best_k}")
    logger.info(classification_report(y, y_pred, digits=5))


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError(
                "Usage: python knesset_word2vec_classification.py <corpus.jsonl> <word2vec.model>"
            )

        corpus_path = sys.argv[1]
        model_path = sys.argv[2]

        logger.info("Loading Word2Vec model...")
        try:
            w2v_model = Word2Vec.load(model_path)
            word_vectors = w2v_model.wv
        except Exception as e:
            raise RuntimeError(f"Failed to load Word2Vec model: {e}")

        logger.info("Finding top speakers...")
        try:
            top = TopTwoSpeakers(corpus_path)
            ln1 = top.speaker1.split()[-1]
            ln2 = top.speaker2.split()[-1]
        except Exception as e:
            raise RuntimeError(f"Failed extracting top speakers: {e}")

        logger.info("Finding speaker name variations...")
        try:
            aliases1, aliases2 = get_variations(corpus_path, ln1, ln2)
        except Exception as e:
            raise RuntimeError(f"Failed extracting speaker variations: {e}")

        task = BinarySentenceClassificationTask(aliases1, aliases2)

        logger.info("Loading and preparing dataset...")
        task.load_data(corpus_path)
        task.balance()
        task.create_sentence_embeddings(word_vectors)

        logger.info("Running KNN classification...")
        run_knn(task)

        logger.info("Program finished successfully")

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
