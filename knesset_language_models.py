import json
import string
import math
from collections import Counter, defaultdict
import pandas as pd
import os
import sys
import random

class ReadJSONLData:
    """
    Loads sentences from JSONL and creates:
    - tokenized sentences (with punctuation)
    - tokenized sentences (no punctuation)
    """
    def __init__(self, jsonl_path):
        self.df = pd.DataFrame()
        self._translator = str.maketrans("", "", string.punctuation)
        self._load(jsonl_path)

    def _load(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sentence = obj.get("sentence_text", "")
                protocol = obj.get("protocol_name", "")  # Changed from protocol_file to protocol_name
                tokens = sentence.split()
                no_punc = [t.translate(self._translator) for t in tokens]
                no_punc = [t for t in no_punc if t]
                data.append({
                    'protocol': protocol,
                    'tokens': tokens,
                    'no_punc_tokens': no_punc,
                    'sentence': sentence
                })
        self.df = pd.DataFrame(data)

class Trigram_LM:
    def __init__(self, sentences):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        # Start + End tokens
        self.vocab = {"s_0", "s_1", "s_2"}
        self.total_unigrams = 0
        self.V = 0
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
            # Add start+end tokens
            padded = ["s_0", "s_1"] + tokens + ["s_2"]
            self.vocab.update(padded)
            for i, w in enumerate(padded):
                self.unigram[w] += 1
                if i >= 1:
                    self.bigram[padded[i - 1]][w] += 1
                if i >= 2:
                    prev2 = padded[i - 2]
                    prev1 = padded[i - 1]
                    self.trigram[(prev2, prev1)][w] += 1
        # FIX: correct unigram total
        self.total_unigrams = sum(self.unigram.values())
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
        padded = ["s_0", "s_1"] + tokens + ["s_2"]
        logp = 0.0
        for i in range(2, len(padded)):
            p = self._interp_prob(padded[i - 2], padded[i - 1], padded[i])
            logp += math.log(p)
        return logp

    def generate_next_token(self, prefix: str):
        """Return (best_token, log_prob) using only real words (no dummy tokens)."""
        tokens = prefix.strip().split()
        padded = ["s_0", "s_1"] + tokens
        w2 = padded[-2]
        w1 = padded[-1]
        best_word = None
        best_prob = -1
        # Only consider real words from training vocab (exclude s_0, s_1, s_2)
        real_words = self.vocab - {"s_0", "s_1", "s_2"}
        for w in real_words:
            p = self._interp_prob(w2, w1, w)
            if p > best_prob:
                best_prob = p
                best_word = w
        return best_word, math.log(best_prob)

class LM_APP:
    def __init__(self, jsonData: ReadJSONLData):
        self.jsonData = jsonData

    def build_models(self):
        self.full = Trigram_LM(self.jsonData.df['tokens'].tolist())
        self.no_punc = Trigram_LM(self.jsonData.df['no_punc_tokens'].tolist())

    def calculate_prob_of_sentence(self, sentence_str: str, use_punc=False):
        model = self.full if use_punc else self.no_punc
        return model.calculate_prob_of_sentence(sentence_str)

    def generate_next_token(self, prefix_str: str, use_punc=False):
        model = self.full if use_punc else self.no_punc
        return model.generate_next_token(prefix_str)

    def get_k_n_t_collocations(self, k: int, n: int, t: int, corpus: pd.DataFrame, type_: str):
        """
        input: k - number of collocations
               n - length of collocation
               t - threshold for the minimum appearance of the collocation in the corpus
               type_ - "frequency" or "tfidf"
        """
        # Prepare documents: dict protocol -> list of tokens
        documents = {}
        for protocol, group in corpus.groupby('protocol'):
            tokens = []
            for tok_list in group['tokens']:
                tokens.extend(tok_list)
            if tokens:
                documents[protocol] = tokens

        D = len(documents)
        if D == 0:
            return []

        # Precompute n-gram counts per doc
        doc_ngram_counts = {}
        doc_total_ng_positions = {}
        for doc, tokens in documents.items():
            l = len(tokens)
            if l < n:
                doc_total_ng_positions[doc] = 0
                doc_ngram_counts[doc] = Counter()
                continue
            doc_total_ng_positions[doc] = l - n + 1
            c = Counter(tuple(tokens[i:i + n]) for i in range(l - n + 1))
            doc_ngram_counts[doc] = c

        # Total n-gram counts across corpus
        total_ngram_counts = Counter()
        for c in doc_ngram_counts.values():
            total_ngram_counts.update(c)

        # Candidates: n-grams with total count >= t
        candidates = [ng for ng, cnt in total_ngram_counts.items() if cnt >= t]

        if not candidates:
            return []

        if type_ == "frequency":
            # Sort by total frequency descending
            sorted_candidates = sorted(
                candidates, key=lambda ng: total_ngram_counts[ng], reverse=True
            )[:k]
            return [' '.join(ng) for ng in sorted_candidates]

        elif type_ == "tfidf":
            # Compute average TF-IDF for each candidate
            ngram_tfidf = {}
            for ng in candidates:
                # DF: number of docs containing ng
                df = sum(1 for c in doc_ngram_counts.values() if ng in c)
                idf = math.log(D / df) if df > 0 else 0
                # Sum TF over all docs
                sum_tf = 0
                for doc in documents:
                    count = doc_ngram_counts[doc][ng]
                    total = doc_total_ng_positions[doc]
                    tf = count / total if total > 0 else 0
                    sum_tf += tf
                avg_tf = sum_tf / D
                avg_tfidf = avg_tf * idf
                ngram_tfidf[ng] = avg_tfidf

            # Sort by avg_tfidf descending
            sorted_ng = sorted(ngram_tfidf, key=ngram_tfidf.get, reverse=True)[:k]
            return [' '.join(ng) for ng in sorted_ng]

        else:
            raise ValueError("Invalid type: must be 'frequency' or 'tfidf'")

    def mask_tokens_in_sentences(self, sentences: list, x: int):
        """Return masked sentences according to mask_percentage."""
        masked = []

        for sent in sentences:
            tokens = sent.split()
            L = len(tokens)

            if L == 0:
                masked.append(sent)
                continue

            #  10 => 0.1% 
            k = max(1, int((x / 100) * L))

            # choose random indices
            mask_indices = set(random.sample(range(L), k))

            new_tokens = [
                ("[*]" if i in mask_indices else t)
                for i, t in enumerate(tokens)
            ]

            masked.append(" ".join(new_tokens))

        return masked

    def sample_10_sentences_randomly(self):
        # filter only sentences with >= 5 tokens
        df_filtered = self.jsonData.df[self.jsonData.df['tokens'].apply(len) >= 5]

        # sample exactly 10
        return df_filtered['sentence'].sample(10, random_state=None).tolist()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python knesset_language_models.py <path/to/corpus_file_name.jsonl> <path/to/output_dir>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    jsonData = ReadJSONLData(jsonl_path)
    model = LM_APP(jsonData)
    model.build_models()
    #print(model.generate_next_token("את", use_punc=False))
    #print(model.generate_next_token("את", use_punc=True))

    # model.mask_tokens_in_sentences(["שקשוקה זה לחלשים"], 10)

    sampled_sentences = model.sample_10_sentences_randomly()

    masked_sentences = model.mask_tokens_in_sentences(sampled_sentences, 10)

    orig_path = os.path.join(output_dir, "original_sampled_sents.txt")
    with open(orig_path, "w", encoding="utf-8") as f:
        for sent in sampled_sentences:
            f.write(sent + "\n")

    masked_path = os.path.join(output_dir, "masked_sampled_sents.txt")
    with open(masked_path, "w", encoding="utf-8") as f:
        for sent in masked_sentences:
            f.write(sent + "\n")


    #Prepare corpora
    # corpus_full = jsonData.df[['protocol', 'tokens']].copy()
    # corpus_no_punc = jsonData.df[['protocol', 'no_punc_tokens']].copy()
    # corpus_no_punc = corpus_no_punc.rename(columns={'no_punc_tokens': 'tokens'})

    # # Output file
    # collocations_path = os.path.join(output_dir, "knesset_collocations.txt")

    # with open(collocations_path, 'w', encoding='utf-8') as f:
    #     # Two-gram
    #     f.write("Two-gram collocations:\n")
    #     f.write("Frequency:\n")
    #     f.write("Full corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 2, 10, corpus_full, "frequency")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("No punctuation corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 2, 10, corpus_no_punc, "frequency")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("TF-IDF:\n")
    #     f.write("Full corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 2, 10, corpus_full, "tfidf")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("No punctuation corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 2, 10, corpus_no_punc, "tfidf")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")

    #     # Four-gram
    #     f.write("Four-gram collocations:\n")
    #     f.write("Frequency:\n")
    #     f.write("Full corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 4, 5, corpus_full, "frequency")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("No punctuation corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 4, 5, corpus_no_punc, "frequency")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("TF-IDF:\n")
    #     f.write("Full corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 4, 5, corpus_full, "tfidf")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")
    #     f.write("No punctuation corpus:\n")
    #     colls = model.get_k_n_t_collocations(10, 4, 5, corpus_no_punc, "tfidf")
    #     for c in colls:
    #         f.write(c + "\n")
    #     f.write("\n")