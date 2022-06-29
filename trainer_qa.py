import torch
from src.models.torch_ligthing_model import T5UQALighteningFineTuner
from src.data.data_module import T5DataModule
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import mlflow


MLFLOW_TRACKING_URI = "https://dagshub.com/espoirMur/unsupervised-open-domain-french-question-answering.mlflow"


if __name__ == "__main__":
    ###############################################################################
    ################################ TRAINER ######################################
    ###############################################################################
    ###############################################################################
    base_model_name = "plguillou/t5-base-fr-sum-cnndm"
    parser = ArgumentParser(description="drc-news-qa-lightning")
    parser = Trainer.add_argparse_args(parent_parser=parser)
    parser = T5UQALighteningFineTuner.add_model_specific_args(parser)
    parser = T5DataModule.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    dict_args["pretrained_module_name"] = base_model_name
    data_module = T5DataModule(**dict_args)
    data_module.prepare_data()
    print("done preparing the data")

    model = T5UQALighteningFineTuner(pretrained_model_name_or_path=dict_args.get("pretrained_module_name"), other_args=dict_args)
    print("done created the model")
    early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=True) # should change to exact matches

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path.cwd().joinpath("checkpoints"),
        save_top_k=1,
        verbose=True,
        monitor="f1_score",
        mode="max",
    )
    lr_logger = LearningRateMonitor()

    trainer = Trainer(max_epochs=2,
                      callbacks=[lr_logger, early_stopping, checkpoint_callback],
                      accelerator="auto",)

    # For CPU Training
    experiment_name = 'unsupervised_qa_with_t5_fusion_in_decoder'
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if dict_args["gpus"] is None or int(dict_args["gpus"]) == 0:
        mlflow.pytorch.autolog()
    elif int(dict_args["gpus"]) >= 1 and trainer.global_rank == 0:
        # In case of multi gpu training, the training script is invoked multiple times,
        # The following condition is needed to avoid multiple copies of mlflow runs.
        # When one or more gpus are used for training, it is enough to save
        # the model and its parameters using rank 0 gpu.
        mlflow.pytorch.autolog()
    else:
        # This condition is met only for multi-gpu training when the global rank is non zero.
        # Since the parameters are already logged using global rank 0 gpu, it is safe to ignore
        # this condition.
        trainer.log.info("Active run exists.. ")

    trainer.fit(model, data_module)


