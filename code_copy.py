from pathlib import Path
DATA_PATH = Path.cwd().parent.joinpath("data")
assert DATA_PATH.exists(), "the data path does not exist"
wikipedia_dump = DATA_PATH.joinpath("wikipedia")
for index, file in enumerate(wikipedia_dump.iterdir()):
    with open(file, "r") as buffer:
        docs = []
        for index_, data in enumerate(buffer.readlines()):
            id_ = f"{index}{index_}"
            json_data = json.loads(json.loads(data))
            json_data["id"] = id_
            document = Document.from_dict(json_data)
            docs.append(document)
        document_store.write_documents(docs, duplicate_documents="skip")
        print("done saving the firt batch")
