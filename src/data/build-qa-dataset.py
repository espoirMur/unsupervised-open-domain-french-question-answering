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

import datasets

from dataclasses import dataclass
from typing import ClassVar, Dict

from datasets.features import Features, Sequence, Value
from datasets.tasks.base import TaskTemplate


@dataclass(frozen=True)
class QuestionAnsweringExtractiveMultipleContext(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task: str = "question-answering-extractive"
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "contexts": Sequence({"text": Value("string")}) })
    label_schema: ClassVar[Features] = Features(Value("string"))
    question_column: str = "question"
    context_column: str = "contexts"
    answers_column: str = "answer"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "contexts", self.answer_column: "answer"}


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

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "contexts": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="murhabazi.com",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractiveMultipleContext(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for article in dataset["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
