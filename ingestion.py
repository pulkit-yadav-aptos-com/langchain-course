import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Loading data...")
    loader = TextLoader("MEDIUMBLOG.TXT")
    documents = loader.load()

    print('splitting documents...')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)


    print('embedding documents...')
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print('vectorizing documents...')
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

    print('done!')