import json
import logging
from collections import Counter
from typing import Tuple, Set
import random
import numpy as np


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

class TopTwoSpeakers:
    """Finds the two most common speakers in the corpus."""
    def __init__(self, corpus_path: str):
        self.speaker1, self.speaker2 = self._get_top_two_speakers(corpus_path)

    def _get_top_two_speakers(self, corpus_path) -> Tuple[str, str]:
        """
        Reads a jsonl corpus file and returns the two most common speakers.
        """
        speaker_counter = Counter()

        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
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
    
        top_1_name = most_common[0][0]
        top_2_name = most_common[1][0]
        
        return top_1_name, top_2_name


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
                    
                    if not speaker:
                        continue
                        
                    if last_name_1 in speaker:
                        variations_1.add(speaker)
                    
                    if last_name_2 in speaker:
                        variations_2.add(speaker)
                        
                except json.JSONDecodeError:
                    continue

        logger.info(f"Found {len(variations_1)} variations for {last_name_1}: {variations_1}")
        logger.info(f"Found {len(variations_2)} variations for {last_name_2}: {variations_2}")
        
        return variations_1, variations_2
        
    except FileNotFoundError:
        logger.error(f"File {corpus_path} not found.")
        return set(), set()


class BinaryClassificationTask:
    """Handles loading data for binary classification between two speakers."""
    def __init__(self, speaker1: set, speaker2: set):
        self.speaker1_aliases = speaker1
        self.speaker2_aliases = speaker2
        
        self.records = [] 
        self.labels = [] 

    def load_data(self, corpus_path):
        """
        Loads FULL records only for the two selected speakers.
        """
        count_s1 = 0
        count_s2 = 0
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        speaker = record.get('speaker_name', '').strip()
                        text = record.get('sentence_text', '').strip()
                        
                        if not speaker or not text:
                            continue

                        if speaker in self.speaker1_aliases:
                            self.records.append(record) 
                            self.labels.append(0) 
                            count_s1 += 1
                        
                        elif speaker in self.speaker2_aliases:
                            self.records.append(record) 
                            self.labels.append(1) 
                            count_s2 += 1
                            
                    except json.JSONDecodeError:
                        continue
                        
            logger.info("--- Binary Data Loaded ---")
            logger.info(f"Class 1 (Label 0): {count_s1} records")
            logger.info(f"Class 2 (Label 1): {count_s2} records")
            logger.info(f"Total: {len(self.records)}")
            
        except FileNotFoundError:
            logger.error(f"File {corpus_path} not found.")
    def balance_classes(self):
        """
        Performs random down-sampling to match the size of the smaller class.
        """
        logger.info("--- Balancing Binary Classes ---")
        
        data = list(zip(self.records, self.labels))
        
        class0 = [d for d in data if d[1] == 0]
        class1 = [d for d in data if d[1] == 1]
        
        len0 = len(class0)
        len1 = len(class1)
        
        logger.info(f"Counts before balancing: Class 0={len0}, Class 1={len1}") # 
        
        min_len = min(len0, len1)
        
        class0_down = random.sample(class0, min_len)
        class1_down = random.sample(class1, min_len)
        
        balanced_data = class0_down + class1_down
        random.shuffle(balanced_data)
        
        self.records, self.labels = zip(*balanced_data)
        
        self.records = list(self.records)
        self.labels = list(self.labels)
        
        logger.info(f"Counts after balancing: Class 0={self.labels.count(0)}, Class 1={self.labels.count(1)}") #

class MultiClassClassificationTask:
    """Handles loading data for multi-class classification: Speaker1, Speaker2, and Other."""
    
    def __init__(self, speaker1_aliases, speaker2_aliases):
        self.speaker1_aliases = speaker1_aliases
        self.speaker2_aliases = speaker2_aliases
        
        self.records = [] 
        self.labels = []

    def load_data(self, corpus_path):
        """
        Loads FULL records. 
        """
        logger.info("Loading Multi-Class Data...")
        count_s1 = 0
        count_s2 = 0
        count_other = 0
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        speaker = record.get('speaker_name', '').strip()
                        text = record.get('sentence_text', '').strip()
                        
                        if not speaker or not text:
                            continue

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
                            
                    except json.JSONDecodeError:
                        continue
                        
            logger.info("--- Multi-Class Data Loaded ---")
            logger.info(f"Class 0 (Speaker 1): {count_s1}")
            logger.info(f"Class 1 (Speaker 2): {count_s2}")
            logger.info(f"Class 2 (Other): {count_other}")
            logger.info(f"Total: {len(self.records)}")
            
        except FileNotFoundError:
            logger.error(f"File {corpus_path} not found.")


    def balance_classes(self):
        """
        Balances 3 classes: Speaker 1, Speaker 2, and Other.
        """
        logger.info("--- Balancing Multi-Class Classes ---")
        
        data = list(zip(self.records, self.labels))
        
        class0 = [d for d in data if d[1] == 0]
        class1 = [d for d in data if d[1] == 1]
        class2 = [d for d in data if d[1] == 2] 
        
        lengths = [len(class0), len(class1), len(class2)]
        logger.info(f"Counts before: Class 0={lengths[0]}, Class 1={lengths[1]}, Class 2={lengths[2]}") # 
        
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



class TaskHandler:
    """Main handler to run the classification tasks."""
    def __init__(self, corpus_file: str):
        self.corpus_file = corpus_file
        self.top_speakers = None
        self.binray_task = None
        self.multi_task = None

    def run(self):
        """Executes the task pipeline."""
        top_speakers = TopTwoSpeakers(self.corpus_file)
    
        if top_speakers.speaker1 and top_speakers.speaker2:
    
            ln1 = top_speakers.speaker1.split()[-1]
            ln2 = top_speakers.speaker2.split()[-1]
            
            aliases1, aliases2 = get_variations(self.corpus_file, ln1, ln2)
            

            self.binray_task = BinaryClassificationTask(aliases1, aliases2)
            self.binray_task.load_data(self.corpus_file)
            self.binray_task.balance_classes()
            
            self.multi_task = MultiClassClassificationTask(aliases1, aliases2)
            self.multi_task.load_data(self.corpus_file)
            self.multi_task.balance_classes()


if __name__ == "__main__":
  
    random.seed(42)
    np.random.seed(42)
    
    corpus_file = 'knesset_corpus.jsonl' 

    handler = TaskHandler(corpus_file)
    handler.run()

    