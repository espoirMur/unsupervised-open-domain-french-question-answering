
import torch
from traitlets import default
from src.utils.evaluation_utils import t5_qa_evaluate, prediction_to_csv
import numpy as np
from pathlib import Path
from transformers import AdamW
from lightning_transformers.core.seq2seq.model import Seq2SeqTransformer
from src.models.fusion_in_decoder import FusionInDecoderModel
from argparse import ArgumentParser
from lightning_transformers.core import TaskTransformer
from typing import Any, List, Optional
from lightning_transformers.core.seq2seq.utils import _pad_tensors_to_max_len
from transformers import T5ForConditionalGeneration


class T5UQALighteningFineTuner(Seq2SeqTransformer):
    def __init__(self, *args, downstream_model_type=FusionInDecoderModel, **kwargs):
        self.pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path")
        self.args = kwargs.pop("other_args")
        val_target_max_length = self.args["answer_maxlength"]
        num_beams = self.args["num_beams"]
        super().__init__(downstream_model_type,
                         val_target_max_length=val_target_max_length,
                         num_beams=num_beams, *args, **kwargs)
        self.override_model()

    def override_model(self):
        
        """
        this is a hack to override the model and use the model the same way it was initialize in the main code. 
        """
        del self.model
        based_t5_model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_name_or_path)
        model = FusionInDecoderModel(based_t5_model.config)
        model.load_t5(based_t5_model.state_dict())
        self.model = model
        model.set_checkpoint(self.args["use_checkpoint"]) ## need to find what this is doing
  
    def training_step(self, train_batch, *args, **kwargs):
        """the training step which return the loss for the batch

        Args:
            train_batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        (idx, labels, _, context_ids, context_mask) = train_batch

        train_loss = self.model(
            input_ids=context_ids.cuda(),
            attention_mask=context_mask.cuda(),
            labels=labels.cuda()
        )[0]
        self.log("train_loss", train_loss)
        return train_loss
    
    def validation_step(self, val_batch, *args, **kwargs):
        """the validation step of the model.

        Args:
            val_batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size_length = val_batch[0].size(0)
        
        (_, target_ids, _, context_ids, context_mask) = val_batch
        predicted_strings = self.generate(
            input_ids=context_ids,
            attention_mask=context_mask,
        )
        gold_strings = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        return {"labels": gold_strings,
                "predictions": predicted_strings}
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        max_length = self.val_target_max_length if self.val_target_max_length else self.model.config.max_length
        generated_tokens = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = _pad_tensors_to_max_len(
                model_cfg=self.model.config, tensor=generated_tokens, max_length=max_length
            )
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str
    
    def test_step(self, test_batch, *args, **kwargs):
        """
        this is almost the same as the validation test
        """
        (_, target_ids, _, context_ids, context_mask) = test_batch
        predicted_strings = self.generate(
            input_ids=context_ids,
            attention_mask=context_mask,
        )
        gold_strings = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        return {"labels": gold_strings,
                "predictions": predicted_strings}
    
    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        predictions, labels = [], []
        for output in outputs:
            for label, pred in zip(output['labels'], output['predictions']):
                predictions.append(pred)
                labels.append(label)
        results = t5_qa_evaluate(labels, predictions)
        exact = torch.tensor(results['exact']).detach()
        f1 = torch.tensor(results['f1']).detach()

        log = {
            'exact_matches': exact,       # for monitoring checkpoint callback
            'f1_score': f1,             # for monitoring checkpoint callback
        }
        self.log_dict(log, logger=True, prog_bar=True, on_epoch=True)
    
    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        predictions, labels = [], []
        for output in outputs:
            for label, pred in zip(output['labels'], output['predictions']):
                predictions.append(pred)
                labels.append(label)
        results = t5_qa_evaluate(labels, predictions)
        exact = torch.tensor(results['exact']).detach()
        f1 = torch.tensor(results['f1']).detach()
        log = {
            'exact_matches': exact,       # for monitoring checkpoint callback
            'f1_score': f1,             # for monitoring checkpoint callback
        }
        results_path = Path.cwd().joinpath("models-predictions.csv")
        prediction_to_csv(prediction=predictions, goldlabel=labels, file_name=results_path)
        self.log_dict(log, logger=True, prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "f1_score",
        }
        return [self.optimizer], [self.scheduler]
    
    @property
    def tokenizer(self):
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @property
    def hf_pipeline_task(self) -> str:
        return "question_answering"


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented argument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = add_optimizer_options(parser)
        parser = add_reader_options(parser)
        return parser


def add_optimizer_options(parser):
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--scheduler_steps', type=int, default=None,help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='fixed')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--fixed_lr', action='store_true')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='path to the checkpoint file')
    return parser


def add_reader_options(parser):
    # parser.add_argument('--model_size', type=str, default='base') can put the model name here in the future
    parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
    parser.add_argument('--text_maxlength', type=int, default=600, help='maximum number of tokens in text segments (question+passage)')
    parser.add_argument('--answer_maxlength', type=int, default=15, help='maximum number of tokens used to train the model, no truncation if -1')
    parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
    parser.add_argument('--n_context', type=int, default=5, help="the number of passages for fusion")
    parser.add_argument('--num_beams', type=int, default=4)
    return parser


def print_memory_usage(step=""):
    print(f"the memory when the {step}  is ")
    free_memory, used_memmory = torch.cuda.mem_get_info()
    print("the free memory is ", free_memory / 1024**2)
    print("the used memory is ", used_memmory / 1024**2)
    print(15 * "***")
    print(torch.cuda.memory_summary())
