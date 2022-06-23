from src.models.torch_ligthing_model import T5UQALighteningFineTuner
from src.data.data_module import T5DataModule
from transformers import  T5Tokenizer
from argparse import ArgumentParser
from pytorch_lightning import Trainer


if __name__ == "__main__":
    ###############################################################################
    ################################ TRAINER ######################################
    ###############################################################################
    ###############################################################################
    parser = ArgumentParser()
    parser.add_argument("--layer_1_dim", type=int, default=128)
    tokenizer = T5Tokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
    fine_tuner = T5UQALighteningFineTuner(pretrained_model_name_or_path="plguillou/t5-base-fr-sum-cnndm")
    parser = fine_tuner.add_model_specific_args(parser)
    data_loaders = T5DataModule.add_model_specific_args(parser)
    args = parser.parse_args()

