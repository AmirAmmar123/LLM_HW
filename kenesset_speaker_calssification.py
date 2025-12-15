import json
import logging
from collections import Counter
from typing import Tuple, Set


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
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


if __name__ == "__main__":
    corpus_file = 'knesset_corpus.jsonl' 
    
    top_speakers = TopTwoSpeakers(corpus_file)
    
    if top_speakers.speaker1 and top_speakers.speaker2:
        logger.info("--- Selected Classes ---")
        logger.info(f"Class 1: {top_speakers.speaker1}")
        logger.info(f"Class 2: {top_speakers.speaker2}")


        last_name_1 = top_speakers.speaker1.split()[-1]
        last_name_2 = top_speakers.speaker2.split()[-1]

        speaker1_aliases, speaker2_aliases = get_variations(
            corpus_file, 
            last_name_1=last_name_1, 
            last_name_2=last_name_2
        )
        

        binary_task = BinaryClassificationTask(speaker1_aliases, speaker2_aliases)
        binary_task.load_data(corpus_file)
        

        multi_task = MultiClassClassificationTask(speaker1_aliases, speaker2_aliases)
        multi_task.load_data(corpus_file)