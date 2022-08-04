from config import DOCUNMENT_STORE_DATABASE_URL
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
from pathlib import Path

INDEX_NAME = 'wikipedia-fr'

document_store = FAISSDocumentStore(sql_url=DOCUNMENT_STORE_DATABASE_URL, validate_index_sync=False)

dense_passage_retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="etalab-ia/dpr-question_encoder-fr_qa-camembert",
    passage_embedding_model="etalab-ia/dpr-ctx_encoder-fr_qa-camembert",
    infer_tokenizer_classes=True,
)

index_path = Path.cwd().joinpath("faiss_index",INDEX_NAME)

if __name__ == "__main__":
    document_store.update_embeddings(dense_passage_retriever)
    document_store.save(index_path)