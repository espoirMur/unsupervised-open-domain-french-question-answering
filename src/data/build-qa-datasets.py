import logging
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from secrets import token_hex
from gensim.corpora.csvcorpus import CsvCorpus
from nltk.corpus.reader.util import concat
from gensim.corpora.csvcorpus import CsvCorpus
from gensim.utils import deaccent
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class DrcNewsCorpus(CsvCorpus):
    def __init__(self, fname, has_header=False):
        """

        Parameters
        ----------
        fname : str
            Path to corpus.
        labels : bool
            If True - ignore first column (class labels).

        """
        logger.info("loading corpus from %s", fname)
        self.fname = fname
        self.has_header = has_header
        self.dataset = pd.read_csv(fname, names=["content", "posted_at"])
        self.length = self.dataset.shape[0]

    def replace_point(self, document):
        """replace the point with the wwt.www with space point before tokenizing the document .
        TOdos : this may have a a downside when the point is in the middle of a words
        Args:
            document (_type_): _description_
        """
        result = re.sub(r"(\S)\.(\S)", r"\1 . \2", document)
        return result

    def remove_accents(self, input_str):
        input_without_accent = deaccent(input_str)
        return input_without_accent

    def clean_text(self, text):
        cleaned_text = self.remove_accents(text)
        cleaned_text = self.replace_point(cleaned_text)
        return str(cleaned_text)

    def tokenize_document(self, document):
        """given a document split the document into sentences and return a list of those sentence
    
        Args:
            document (_type_): _description_
    
        Returns:
            _type_: _description_
        """
        tokenized_document = sent_tokenize(document, language="french")
        return tokenized_document
    
    def replace_between(self, text, begin, end, alternative='<MASK>'):
        to_replace = text[begin:end]
        return text.replace(to_replace, alternative)
    
    def build_question_answers_from_sentences(self, sentence, nlp):
        """given a sentence build the question and the answers from the sentence.

        Args:
            sentence (_type_): _description_

        Returns:
            _type_: _description_
        """
        entities = nlp(sentence)
        for entity in entities:

            start_index = entity.get("start")
            end_index = entity.get("end")
            token = entity.get("word")
            entity_group = entity.get("entity_group")
            entity = sentence[start_index:end_index]
            sentence_with_mask = self.replace_between(sentence, start_index, end_index, alternative=' <MASK> ')
            yield (sentence_with_mask, entity, entity_group)
    
    def save_question_answer_to_file(self, output_file_path, n_sample=None):
        """
        generate the json representation of the corpus and save it a a loadable version in csv
        Args:
            document_series (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(output_file_path, "w") as output_file:
            for item in self.__iter__(n_sample=n_sample):
                sentence = item.get("text")
                date = item.get("posted_at")
                for sentence_with_mask, entity, entity_group in self.build_question_answers_from_sentences(sentence, nlp):
                    line_to_save = ' | '.join([sentence_with_mask, entity, entity_group ] +[ date, "\n"])
                    output_file.write(line_to_save)
            
    def __iter__(self, n_sample=None):
        """Iterate over the corpus,  a line  at a time

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        if n_sample:
            self.dataset = self.dataset.sample(n_sample)
        
        for _, line in tqdm(self.dataset.sample(frac=1).reset_index(drop=True).fillna("").iterrows(), total=self.length):
            cleaned_line = self.clean_text(line["content"])
            for sentence in self.tokenize_document(cleaned_line):
                posted_at = line["posted_at"] if line["posted_at"] else ""
                item = {"_task_hash": token_hex(6), "_input_hash": token_hex(6),  "text": sentence,  "posted_at": posted_at}
                yield item
