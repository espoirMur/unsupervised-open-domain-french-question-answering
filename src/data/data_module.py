from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer
from src.data.torch_datasets import Dataset, Collator
from torch.utils.data import DataLoader
from argparse import ArgumentParser


class T5DataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super(T5DataModule, self).__init__()
        self.pre_trained_module_name = kwargs.get('pre_trained_module_name')
        self.train_dataset_path = kwargs.get('train_dataset_path')
        self.val_dataset_path = kwargs.get('val_dataset_path')
        self.test_dataset_path = None
        self.max_question_length = None
        self.max_passage_length = None
        self.max_answer_length = None
        self.tokenizer = None
        self.args = kwargs
        self.RANDOM_SEED = 42
    
    def prepare_data(self) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(self.pre_trained_module_name)
        self.collator = Collator(self.text_maxlength, self.tokenizer, answer_maxlength=self.max_answer_length)

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
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_dataset_path', type=str, default='none', help='path of train data')
        parser.add_argument('--eval_dataset_path', type=str, default='none', help='path of eval data')
        return parser

    def generate_data_loaders(self, split) -> DataLoader:
        """create the test set dataLoaders
        either for train, val, or test

        """
        dataset_path = getattr(self, f"{split}_dataset_path")
        test_dataset = Dataset(dataset_path,
                               self.max_question_length,
                               self.max_passage_length,
                               self.max_answer_length)
        return DataLoader(test_dataset, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"])
    
    def train_data_loader(self) -> DataLoader:
        return self.generate_data_loaders("train")
    
    def val_data_loader(self) -> DataLoader:
        return self.generate_data_loaders("val")
    
    def test_data_loader(self) -> DataLoader:
        return self.generate_data_loaders("test")
