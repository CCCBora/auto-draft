from langchain.embeddings import HuggingFaceEmbeddings


model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

all_minilm_l6_v2 = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)


EMBEDDINGS = {"all-MiniLM-L6-v2": all_minilm_l6_v2}