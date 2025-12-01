import json
import string
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

class ReadJSONLData:
    """
    Class to read JSONL data and store sentences and metadata.
    Attributes:
        sentences: List of raw sentence strings.
        sentences_tokens: List of tokenized sentences (with punctuation).
        sentences_no_punct: List of tokenized sentences (without punctuation).
        metadata: List of metadata dictionaries for each sentence.
        N: Corpus size.
        V_with_punct: Vocabulary size including punctuation.
        V_no_punct: Vocabulary size excluding punctuation.
        unique_tokens: Set of unique tokens (with punctuation).
        unique_tokens_no_punct: Set of unique tokens (without punctuation).
        token_counter: Counter of tokens with punctuation.
        token_counter_no_punct: Counter of tokens without punctuation.
    """
    def __init__(self, jsonl_path):
        self.sentences = []
        self.sentences_tokens = []
        self.sentences_no_punct = []
        self.metadata = []
        self.all_tokens_joined = []
        self.N = 0
        self.V_with_punct = 0
        self.V_no_punct = 0
        self.unique_tokens = set()
        self.unique_tokens_no_punct = set()
        self.token_counter = Counter()
        self.token_counter_no_punct = Counter()

        self._translator = str.maketrans('', '',  string.punctuation )
        self._load_jsonl(jsonl_path)

    def _read_jsonl_lines(self, path):
        """Generator to read JSONL line by line."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line.strip())
        except FileNotFoundError:
            raise ValueError(f"JSONL file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL format: {e}")

    def _load_jsonl(self, path):
        """Load JSONL, tokenize sentences, and compute vocab statistics."""
        for obj in self._read_jsonl_lines(path):
            self.metadata.append(obj)
            sentence = obj.get("sentence_text", "")
            if not sentence.strip():
                continue  # Skip empty sentences
            self.sentences.append(sentence)
            tokens = sentence.split()  # Simple split; assume tokens separated by space
            self.sentences_tokens.append(tokens)
            self.token_counter.update(tokens)
            # Remove punctuation: translate and filter non-empty
            no_punct_tokens = [tok.translate(self._translator) for tok in tokens 
                               if tok.translate(self._translator).strip()]
            self.sentences_no_punct.append([t for t in no_punct_tokens if t])  # Filter empty after translate
            self.token_counter_no_punct.update(no_punct_tokens)
        self.N = sum(self.token_counter.values())
        self.V_with_punct = len(self.token_counter)
        self.V_no_punct = len(self.token_counter_no_punct)
        self.unique_tokens = set(self.token_counter.keys())
        self.unique_tokens_no_punct = set(self.token_counter_no_punct.keys())
        self.all_tokens_joined = [tok for sent in self.sentences_tokens for tok in sent]


class TrigramModel:

    def __init__(self, tokenized_sentences):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)

        # INTERNAL tokens only
        self.vocab = {"s_0", "s_1"}
        self.total_unigrams = 0
        self.V = 0

        self.lambdas = {"tri": 0.6, "bi": 0.25, "uni": 0.15}

        self._build(tokenized_sentences)

    # ----------------------------------------------------------
    def _build(self, sentences):
        for sent in sentences:
            if not sent:
                continue

            padded = ["s_0", "s_1"] + sent 
            self.vocab.update(padded)

            for i, w in enumerate(padded):
                # unigram
                self.unigram[w] += 1
                self.total_unigrams += 1

                # bigram
                if i >= 1:
                    prev = padded[i - 1]
                    self.bigram[prev][w] += 1

                # trigram
                if i >= 2:
                    prev2, prev1 = padded[i - 2], padded[i - 1]
                    self.trigram[(prev2, prev1)][w] += 1

        self.V = len(self.vocab)

    # ----------------------------------------------------------
    def _laplace(self, count, denom):
        return (count + 1) / (denom + self.V)

    # ----------------------------------------------------------
    def _interp_prob(self, w2, w1, w):
        # trigram
        tri_count = self.trigram[(w2, w1)][w]
        tri_denom = sum(self.trigram[(w2, w1)].values())
        p_tri = self._laplace(tri_count, tri_denom)

        # bigram
        bi_count = self.bigram[w1][w]
        bi_denom = sum(self.bigram[w1].values())
        p_bi = self._laplace(bi_count, bi_denom)

        # unigram
        uni_count = self.unigram[w]
        p_uni = self._laplace(uni_count, self.total_unigrams)

        return (self.lambdas["tri"] * p_tri +
                self.lambdas["bi"]  * p_bi  +
                self.lambdas["uni"] * p_uni)

    # ----------------------------------------------------------
    def calculate_prob_of_sentence(self, sentence):
        tokens = sentence.split()
        if not tokens:
            return float("-inf")

        padded = ["s_0", "s_1"] + tokens

        logp = 0.0
        for i in range(2, len(padded)):
            p = self._interp_prob(padded[i - 2], padded[i - 1], padded[i])
            logp += math.log(p)

        return logp

    # ----------------------------------------------------------
    def generate_next_token(self, prefix):
        tokens = prefix.strip().split()
        padded = ["s_0", "s_1"] + tokens

        w2, w1 = padded[-2], padded[-1]

        best_word = None
        best_prob = -1

        for w in self.vocab:
            # DO NOT expose internal tokens to users
            if w in {"s_0", "s_1"}:
                continue

            p = self._interp_prob(w2, w1, w)
            if p > best_prob:
                best_prob = p
                best_word = w

        return best_word, math.log(best_prob)


class Trigram_LM:
    def __init__(self, jsonData: ReadJSONLData):
        self.full = TrigramModel(jsonData.sentences_tokens)
        self.no_punc = TrigramModel(jsonData.sentences_no_punct)

    def calculate_prob_of_sentence(self, sentence, use_no_punc=False):
        model = self.no_punc if use_no_punc else self.full
        return model.calculate_prob_of_sentence(sentence)

    def generate_next_token(self, prefix:str, use_no_punc=False):
        model = self.no_punc if use_no_punc else self.full
        return model.generate_next_token(prefix)
    
    

if __name__ == "__main__":
    data_path = "./knesset_corpus.jsonl"
    jsonData = ReadJSONLData(data_path)

    model = Trigram_LM(jsonData)
    print(model.generate_next_token("על מה אני"))