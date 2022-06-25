
import torch
from src.utils.evaluation_utils import ems, weighted_average
import numpy as np
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
            input_ids=context_ids.to(self.device),
            attention_mask=context_mask.to(self.device),
            labels=labels.to(self.device)
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
        three_random_indexes = np.random.choice(batch_size_length, 3, replace=False)
        (_, target_ids, _, context_ids, context_mask) = val_batch
        predicted_strings = self.generate(
            input_ids=context_ids.to(self.device),
            attention_mask=context_mask.to(self.device),
        )
        gold_strings = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        exact_matches = [ems(predicted_string, [gold_string]) for predicted_string, gold_string in zip      (predicted_strings, gold_strings)]
        total = len(exact_matches)
        exact_matches = torch.tensor(exact_matches, dtype=torch.float16).to(self.device)
        total = exact_matches.size(0)
        sample_predictions = [predicted_strings[i] for i in three_random_indexes]
        sample_gold_strings = [gold_strings[i] for i in three_random_indexes]
        self.log("exact_matches", exact_matches.sum(), prog_bar=True)
        return {'exact_matches': exact_matches, "total": total, "sample_predictions": sample_predictions,   "sample_gold_strings": sample_gold_strings}
    
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
    
    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        totals = sum([output["total"] for output in outputs])
        exact_matches = [output["exact_matches"].mean() for output in outputs]
        sample_predictions = [output["sample_predictions"] for output in outputs]
        sample_gold_strings = [output["sample_gold_strings"] for output in outputs]
        for sample_prediction, sample_gold_string in zip(sample_predictions, sample_gold_strings):
            self.print(f"sample gold strings: {sample_gold_string} ===== sample predictions: {sample_prediction}")
        self.print("the sample matches are ", exact_matches)
        exact_matches, total = weighted_average(np.mean(exact_matches),
                                                totals, {"device": self.device,
                                                         "is_distributed": self.args.get("is_distributed", False)})
        self.log("exact_matches", exact_matches)
        self.log("total", total)
    
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
            "monitor": "exact_matches",
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
    return parser


def add_reader_options(parser):
    # parser.add_argument('--model_size', type=str, default='base') can put the model name here in the future
    parser.add_argument('--use_checkpoint', action='store_false', help='use checkpoint in the encoder')
    parser.add_argument('--text_maxlength', type=int, default=500, help='maximum number of tokens in text segments (question+passage)')
    parser.add_argument('--answer_maxlength', type=int, default=15, help='maximum number of tokens used to train the model, no truncation if -1')
    parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
    parser.add_argument('--n_context', type=int, default=10)
    parser.add_argument('--num_beams', type=int, default=4)
    return parser
 