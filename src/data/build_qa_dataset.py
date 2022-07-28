# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""PIAF Question Answering Dataset"""


import json
from multiprocessing import Value

import datasets
from haystack.schema import Document
from dataclasses import dataclass
from typing import ClassVar, Dict
from datasets.tasks.base import TaskTemplate
from pathlib import Path
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
To be filled once the paper is out
"""

_DESCRIPTION = """\
this is athe dataset I build using my work for unsupervised question answering
"""

_URLS = {"train": "https://github.com/etalab-ia/piaf-code/raw/master/piaf-v1.0.json"}


class UnsupervisedQuestionAnswersConfig(datasets.BuilderConfig):
    """BuilderConfig for PIAF."""

    def __init__(self, **kwargs):
        """BuilderConfig for PIAF.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UnsupervisedQuestionAnswersConfig, self).__init__(**kwargs)


class UnsupervisedQuestionAnswersConfig(datasets.GeneratorBasedBuilder):
    """The Piaf Question Answering Dataset. Version 1.0."""

    BUILDER_CONFIGS = [
        UnsupervisedQuestionAnswersConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    DATA_PATH = Path.cwd().joinpath("data")
    BASE_QA_PATH = DATA_PATH.joinpath("processed", "DRC-News-UQA")
    assert BASE_QA_PATH.exists()
    qa_dataset_file = BASE_QA_PATH.joinpath("drc-news-uqa-small.json")
    assert qa_dataset_file.exists()
    files_urls = {"train": qa_dataset_file}

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.features.Value("string"),
                    "title": datasets.features.Value("string"),
                    "answer": datasets.features.Value("string"),
                    "question": datasets.features.Value("string"),
                    "contexts": datasets.Sequence(feature={"content": datasets.features.Value("string"),
                                                           "posted_at": datasets.features.Value("string"),
                                                           "title": datasets.features.Value("string")}),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and contexts as input).
            supervised_keys=None,
            homepage="murhabazi.com",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = self.qa_dataset_file
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for question in dataset:
                id = question.get("id")
                yield id, question
