import json
import string
import math
from collections import Counter, defaultdict


class ReadJSONLData:
    """
    Loads sentences from JSONL and creates:
    - tokenized sentences (with punctuation)
    - tokenized sentences (no punctuation)
    """

    def __init__(self, jsonl_path):
        self.sentences = []
        self.sentences_tokens = []
        self.sentences_no_punct = []

        self._translator = str.maketrans("", "", string.punctuation)
        self._load(jsonl_path)

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sentence = obj.get("sentence_text", "")

                tokens = sentence.split()
                self.sentences_tokens.append(tokens)

                no_punc = [t.translate(self._translator) for t in tokens]
                no_punc = [t for t in no_punc if t]
                self.sentences_no_punct.append(no_punc)

                self.sentences.append(sentence)



class TrigramModel:

    def __init__(self, sentences):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)


        self.vocab = {"s_0", "s_1", "s_end"}

        self.total_unigrams = 0
        self.V = 0

        # Interpolation weights (chosen arbitrarily; explained in report)
        self.lambdas = {
            "tri": 0.6,
            "bi": 0.25,
            "uni": 0.15
        }

        self._build_model(sentences)

    def _build_model(self, sentences):
        for tokens in sentences:
            if not tokens:
                continue

            # Add dummy start tokens and end token (internal only)
            padded = ["s_0", "s_1"] + tokens + ["s_end"]

            # Expand vocabulary
            self.vocab.update(padded)

            # Build counts
            for i, w in enumerate(padded):
                self.unigram[w] += 1
                self.total_unigrams += 1

                if i >= 1:
                    self.bigram[padded[i - 1]][w] += 1

                if i >= 2:
                    prev2 = padded[i - 2]
                    prev1 = padded[i - 1]
                    self.trigram[(prev2, prev1)][w] += 1

        self.V = len(self.vocab)

    
    def _laplace(self, count, denom):
        return (count + 1) / (denom + self.V)


    def _interp_prob(self, w2, w1, w):
        # Trigram
        tri_count = self.trigram[(w2, w1)][w]
        tri_denom = sum(self.trigram[(w2, w1)].values())
        p_tri = self._laplace(tri_count, tri_denom)

        # Bigram
        bi_count = self.bigram[w1][w]
        bi_denom = sum(self.bigram[w1].values())
        p_bi = self._laplace(bi_count, bi_denom)

        # Unigram
        uni_count = self.unigram[w]
        p_uni = self._laplace(uni_count, self.total_unigrams)

        # Interpolation
        return (
            self.lambdas["tri"] * p_tri +
            self.lambdas["bi"]  * p_bi  +
            self.lambdas["uni"] * p_uni
        )

    def calculate_prob_of_sentence(self, sentence: str) -> float:
        """Return log probability of sentence (float)."""

        tokens = sentence.strip().split()
        if not tokens:
            return float("-inf")

        padded = ["s_0", "s_1"] + tokens + ["s_end"]

        logp = 0.0
        for i in range(2, len(padded)):
            p = self._interp_prob(padded[i - 2], padded[i - 1], padded[i])
            logp += math.log(p)

        return logp

    def generate_next_token(self, prefix: str):
        """Return (best_token, log_prob)."""

        tokens = prefix.strip().split()
        padded = ["s_0", "s_1"] + tokens

        w2 = padded[-2]
        w1 = padded[-1]

        best_word = None
        best_prob = -1

        for w in self.vocab:
            # Do NOT return dummy tokens
            if w in {"s_0", "s_1", "s_end"}:
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

    def calculate_prob_of_sentence(self, sentence_str: str, use_punc=False):
        model = self.full if use_punc else self.no_punc
        return model.calculate_prob_of_sentence(sentence_str)

    def generate_next_token(self, prefix_str: str, use_punc=False):
        model = self.full if use_punc else self.no_punc
        return model.generate_next_token(prefix_str)
    


if __name__ == "__main__":
    data_path = "./knesset_corpus.jsonl"
    jsonData = ReadJSONLData(data_path)

    model = Trigram_LM(jsonData)
    print(model.generate_next_token("ראש הממשלה", False))