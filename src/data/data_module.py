from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer
from src.data.torch_datasets import Dataset, Collator
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path


class T5DataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super(T5DataModule, self).__init__()
        self.pretrained_module_name = kwargs.get('pretrained_module_name')
        self.train_dataset_path = kwargs.get('train_dataset_path')
        self.val_dataset_path = kwargs.get('val_dataset_path')
        self.test_dataset_path = None
        self.text_maxlength = kwargs.get('text_maxlength')
        self.answer_maxlength = kwargs.get('answer_maxlength')
        self.n_context = kwargs.get('n_context')
        self.tokenizer = None
        self.args = kwargs
        self.RANDOM_SEED = 42
    
    def prepare_data(self) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_module_name)
        self.collator = Collator(self.text_maxlength, self.tokenizer, answer_maxlength=self.answer_maxlength)

    def setup(self, stage: str) -> None:
        """Not using this was supposed to split the data into train, val, and test.

        Args:
            stage (str): _description_
        """
        pass
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented argument parser
        """
        train_dataset_path, val_dataset_path = generate_dataset_file_names()
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_dataset_path', type=str, default=train_dataset_path, help='path of train data')
        parser.add_argument('--val_dataset_path', type=str, default=val_dataset_path, help='path of eval data')
        parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--num_workers", default=8, type=int, help="Number of workers for data loading.")

        return parser

    def generate_data_loaders(self, split) -> DataLoader:
        """create the test set dataLoaders
        either for train, val, or test

        """
        dataset_path = getattr(self, f"{split}_dataset_path")
        dataset = Dataset(dataset_path,
                          self.n_context)
        return DataLoader(dataset,
                          batch_size=self.args["batch_size"],
                          num_workers=self.args["num_workers"],
                          collate_fn=self.collator,)
    
    def train_dataloader(self) -> DataLoader:
        return self.generate_data_loaders("train")
    
    def val_dataloader(self) -> DataLoader:
        return self.generate_data_loaders("val")
    
    def test_dataloader(self) -> DataLoader:
        return self.generate_data_loaders("test")


def generate_dataset_file_names() -> None:
    """
    This function reads the data and generates the data files
    """
    DATA_PATH = Path.cwd().joinpath("data")
    BASE_QA_PATH = DATA_PATH.joinpath("processed", "DRC-News-UQA")
    assert BASE_QA_PATH.exists()
    train_dataset_file = BASE_QA_PATH.joinpath("drc-news-uqa-small.json")
    val_dataset_file = BASE_QA_PATH.joinpath("drc-news-uqa-small-dev.json")
    assert train_dataset_file.exists()
    assert val_dataset_file.exists()
    return train_dataset_file.__str__(), val_dataset_file.__str__()
