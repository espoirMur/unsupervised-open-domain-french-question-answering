import torch
import random
import json
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 n_context=None,
                 question_prefix="question:",
                 passage_prefix="context:"):
        self.data_path = data_path
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.passage_prefix = passage_prefix
        self.data = self._load_data(data_path)
    
    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        return f"{example['answer']} </s>"

    def __getitem__(self, index):
        example = self.data[index]
        question_ = example["question"].replace('<MASK>', '')  # replace because we wrongly put mask here, we can remove it and it will work.
        question = f"{self.passage_prefix} {question_} </s>"
        target = self.get_target(example)
        contexts = example["contexts"][:self.n_context]
        passages = [f'{self.passage_prefix} {context["content"]}' for context in contexts]
        # the current data does not have the score , we put it to the position of the data
        scores = [1.0 / (index + 1) for index in range(len(contexts))]
        return {"index": index,
                "question": question,
                "passages": passages,
                "target": target,
                "scores": scores}

    def get_example(self, index):
        return self.data[index]
    
    def _load_data(self, filepath):
        """This function returns the examples in the raw (text) form.
        it will return the content of the list as the examples
        """
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            return dataset


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=100):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] is not None)
        index = torch.tensor([example['index'] for example in batch])
        target = [example['target'] for example in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = self.encode_passages(text_passages)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

    def encode_passages(self, batch_text_passages):
        passage_ids, passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = self.tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.text_maxlength,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )
            passage_ids.append(p['input_ids'][None])
            passage_masks.append(p['attention_mask'][None])

        passage_ids = torch.cat(passage_ids, dim=0)
        passage_masks = torch.cat(passage_masks, dim=0)
        return passage_ids, passage_masks.bool()
