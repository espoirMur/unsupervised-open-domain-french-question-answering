import json
from tqdm import tqdm
from secrets import token_hex
from haystack.nodes import BM25Retriever


class Entity:
    def __init__(self, name, start, end, group):
        self.name = name
        self.start = start
        self.end = end
        self.group = group

    @staticmethod
    def from_dict(cls, dict):
        """generate an entity from a dict
        Args:

        Args:
            dict (_type_): _description_

        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        name = dict.get("name")
        start = dict.get("start")
        end = dict.get("end")
        group = dict.get("group")
        return cls(name, start, end, group)
    

class Sentence:
    def __init__(self, text):
        self.text = text
    
    def generate_question_answer_from_entity(self, entity):
        entity_start = entity.start
        entity_end = entity.end
        return self.text[:entity_start] + " <MASK> " + self.text[entity_end:], self.text[entity_start:entity_end]

    def generate_question_answers(self):
        if not getattr(self, "entities", None):
            raise Exception("Call the function build entities to generate entities")
        for entity in self.entities:
            yield self.generate_question_answer(entity.start, entity.end)
    
    def get_search_query(self, start, end):
        return self.text[:start] + " " + self.text[end:]
    
    def get_answer(self, entity_start, entity_end):
        return self.text[entity_start:entity_end]
    
    def get_hard_negative_context(self, retriever: BM25Retriever, n_ctxs: int = 15, entity: Entity = None):
        """given an entity get the hard negative context

        Args:
            retriever (BM25Retriever): _description_
            n_ctxs (int, optional): _description_. Defaults to 15.
            entity (Entity, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        entity_start = entity.start
        entity_end = entity.end
        answer = self.get_answer(entity_start, entity_end)
        search_query = self.get_search_query(entity_start, entity_end)
        list_hard_neg_ctxs = []
        retrieved_docs = retriever.retrieve(query=search_query, top_k=n_ctxs, filters={"name_to_not_match": answer})
        for index, retrieved_doc in enumerate(retrieved_docs):
            retrieved_doc_text = retrieved_doc.content
            if answer.lower() in retrieved_doc_text.lower():
                continue
            list_hard_neg_ctxs.append(
                {"title": f"document_{index}", "text": retrieved_doc_text, "posted_at": retrieved_doc.meta.get("posted_at")}
            )

        return list_hard_neg_ctxs
    
    def get_positive_context(self, retriever: BM25Retriever, positive_documents: int = 100, entity: Entity = None):
        """given entitity retrieve the positive context
        we will first retrieve the top  100 documents , 
        - if the answer is in the top 40 document the input of the reader is the top 40 documents
        if the top 40 documents does not contain the answer we check whithin the top 41 to 100 document if the anwer is ther and we put it ther.
        other wise we discard the sentence

        Args:
            retriever (BM25Retriever): _description_
            n_ctxs (int, optional): _description_. Defaults to 15.
            entity (Entity, optional): _description_. Defaults to None.
        """
        search_query, answer = self.generate_search_query_from_entity(entity)
        list_pos_ctxs = []
        retrieved_docs = retriever.retrieve(query=search_query, top_k=positive_documents)
        for index, retrieve_doc in enumerate(retrieved_docs[0:40]):
            if answer.lower() in retrieve_doc.content.lower():
                list_pos_ctxs.append(
                    {"title": f"document_{index}", "content": retrieve_doc.content, "posted_at": retrieve_doc.meta.get("posted_at")}
                )
        if len(list_pos_ctxs) == 0:
            for index, retrieve_doc in enumerate(retrieved_docs[40:100]):
                if answer.lower() in retrieve_doc.content.lower():
                    list_pos_ctxs.append(
                        {"title": f"document_{index}", "content": retrieve_doc.content, "posted_at": retrieve_doc.meta.get("posted_at")}
                    )
        else:
            pass
        if len(list_pos_ctxs) == 0:
            return []
        return list_pos_ctxs

    def build_entities(self, ner_pipeline):
        """given a sentence generate names entities

        Args:
            ner_pipeline (_type_): _description_
        """
        entities = ner_pipeline(self.text)
        filtered_entities = self.filter_entities(entities)
        self.entities = [Entity.from_dict(Entity, entity) for entity in filtered_entities]

    def filter_entities(self, entities):
        """filter the entities and keep only name , org, loc, date and the entity with a score of more than 85%

        Args:
            entities (_type_): _description_
        """
        return [entity for entity in entities if entity.get("entity_group") in ["PER", "ORG", "LOC", "DATE"] and entity.get("score") >= 0.85]
    
    def to_squad_format(self, entity, bm25_retriever_positive):

        """given a sentence generate a squad format"""
        question, answer = self.generate_question_answer_from_entity(entity)
        id_ = token_hex(4)
        positive_retrieved_docs = self.get_positive_context(bm25_retriever_positive, entity=entity)
        if positive_retrieved_docs:
            return id_, {
                            "title": " ",
                            "contexts": positive_retrieved_docs,
                            "question": question,
                            "id": id_,
                            "answer": {
                                "answer_start": entity.start,
                                "text": answer,
                            },
                        }
        else:
            return None, None
    
    def to_json_file(self, entity, bm25_retriever_positive, base_folder):
        """save the squad format to json file"""
        id_, squad_format = self.to_squad_format(entity, bm25_retriever_positive)
        if squad_format:
            with open(base_folder.joinpath(f"{id_}.json"), "w") as f:
                json.dump(squad_format, f, indent=4, ensure_ascii=False)

    def generate_search_query_from_entity(self, entity):
        entity_start = entity.start
        entity_end = entity.end
        question = self.get_search_query(entity_start, entity_end)
        answer = self.get_answer(entity_start, entity_end)
        return question, answer


class DocumentContext:
    def __init__(self, content, spacy_pipeline=None, ner_pipeline=None, bm25_retriever_positive=None):
        self.context = content
        self.spacy_pipeline = spacy_pipeline
        self.ner_pipeline = ner_pipeline
        self.positive_retriever = bm25_retriever_positive
        self.spacy_doc = spacy_pipeline(content)
        self.sentences = list()
    
    def generate_sentences(self):
        """sentence is a list of sentence from span from context with different entities

        Args:
            sentences (_type_): _description_
        """
        for sentence in self.split_document_in_sentence():
            sentence_object = Sentence(sentence.text)
            self.sentences.append(sentence_object)
            
    def split_document_in_sentence(self):
        """take the document and yield valid sentences from the document context

        Args:
            doc (_type_): _description_
        """
        for sentence in self.spacy_doc.sents:
            if self.is_valid_sentence(sentence.text):
                yield sentence
    
    def is_valid_sentence(self, sentence):
        """# I am not loosing a lot by using only the sentences which ends as with a dot , question mark or exclamation mark 

        Args:
            sentence (_type_): _description_

        Returns:
            _type_: _description_
        """
        return sentence.endswith((".", "?", "!")) and 40 <= len(sentence) <= 250


class AllCorpusBuilder:
    """the main corpus builder that takes the documents and save the content to the file
    it takes the ner pipeline , the transformer pipeline, and  the retriever 
    """
    def __init__(self, ner_pipeline, transformer_pipeline, retriever, all_docs, base_folder):
        self.ner_pipeline = ner_pipeline
        self.transformer_pipeline = transformer_pipeline
        self.retriever = retriever
        self.all_docs = all_docs
        self.base_folder = base_folder
    
    def build_corpus(self):
        for document in tqdm(self.all_docs, desc="generating dataset"):
            document_context = DocumentContext(document.content, self.ner_pipeline, self.transformer_pipeline, self.retriever)
            document_context.generate_sentences()
            for sentence in document_context.sentences:
                sentence.build_entities(self.transformer_pipeline)
                for entity in sentence.entities:
                    sentence.to_json_file(entity, self.retriever, self.base_folder)
