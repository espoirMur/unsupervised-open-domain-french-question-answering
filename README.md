# Fusion-in-Decoder

This project implements the [fusion in decoder](https://github.com/FenQQQ/Fusion-in-decoder) model for Open Domain Question answering using [PyTorchLighting](https://www.pytorchlightning.ai/).


### What was this project used for?

This project was used to develop an end-to-end framework for Question Answering in French for my master thesis. The framework consisted of two components: a retriever and a reader. The reader is a classic Information Retrieval system with ElasticSearch. It uses a variant of the Term Frequency, Inverse Document Frequency(TF-IDF), called BM25, to retrieve paragraphs related to a query. The reader was a generative transformer model from the T5 family, which use an approach called [fusion in decoder](https://github.com/FenQQQ/Fusion-in-decoder).

Once completed, the model will be published on HuggingFace.

### Installing requirements.

This project is built with Python 3.8.

- Create a virtual environment and make sure it is activated.
- Install the requirements by running the following command: `pip install -r requirements.txt`

### Instructions to train the model.


First, the data must be downloaded; you can download the data with this code.

`put the code to download the data here.`

Run the following command : 

`python trainer_qa.py --batch_size 4 --train_dataset_path  path_to_train/ --val_dataset_path path_to_valid/ --n_context 4 -runner_name one-prarragrah-fquad-bm25`
