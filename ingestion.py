import os
import asyncio
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap, TavilyCrawl
from logger import log_info, log_warning, log_error, log_success, log_header, Colors


load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=3, max_breadth=15, max_pages=50)
tavily_crawl = TavilyCrawl()

async def main():
    """"Main asyncfunction to orchestrate the entire process"""
    log_header("Documentation Ingestion Pipeline")

    log_info("Starting to crawl documentation website",Colors.PURPLE)
    
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 3,
        "extract_depth": "advanced",
    })

    all_docs = [Document(page_content=doc['raw_content'], metadata={"source": doc['url']}) for doc in res["results"]]
    log_success(f"TavilyCrawl found {len(all_docs)} pages from the documentation website")

    log_header("Document Chunking and Embedding")
    log_info(f"Text Splitter : Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Successfully split {len(all_docs)} documents into {len(splitted_docs)} chunks")

    await index_documents(splitted_docs,batch_size=100)

    log_header("PIPELINE COMPLETED SUCCESSFULLY")




async def index_documents(docs: List[Document],batch_size: int = 100):
    """Index documents in batches into Pinecone"""

    log_header("VECTOR STORAGE PHASE")
    log_info(f"Indexing {len(docs)} documents into Pinecone", Colors.DARKCYAN)

    batches = [
        docs[i:i+batch_size] for i in range(0, len(docs), batch_size)
    ]

    log_info(f"Processing {len(batches)} batches of {batch_size} documents", Colors.YELLOW)


    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(f"Batch {batch_num} of {len(batch)} documents indexed successfully")
        except Exception as e:
            log_error(f"Error indexing batch {batch_num}: {e}")
            return False
        return True

    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for result in results if result is True)
    if(success_count == len(batches)):
        log_success(f"All {len(docs)} documents indexed successfully")
    else:
        log_warning(f"Failed to index {len(docs) - success_count} documents")


if __name__ == "__main__":
    asyncio.run(main())