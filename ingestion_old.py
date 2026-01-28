import asyncio
import ssl
import os
import certifi
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import(Colors, log_info, log_success, log_error, log_warning, log_header)

load_dotenv()

#Configure SSL context to use certifi certificates
ss_context = ssl._create_default_https_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("INDEX_NAME"))
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=2, max_breath=20 ,max_pages=10)
tavily_crawl = TavilyCrawl()

#loader = TextLoader("./index.md")

async def main():
    """Main ingestion pipeline."""
    log_header("Doc Helper Ingestion Pipeline")

    log_info("TavilyCrawl..Starting to crawl documentation from http://python.langchain.com/en/latest/)", Colors.PURPLE)

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "allowed_domains": ["python.langchain.com"],
            "max_depth": 5,
            "extract_depth": "advanced",
            "instructions" : "content on ai agents, chat models, and langchain usage"
        }
    )

    all_docs = [
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}
     )
        for result in res["results"]
    ]    

    log_success(f"Crawled and extracted {len(all_docs)} document(s) from Langchain documentation.")

    log_info("Splitting documents into chunks...", Colors.YELLOW)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    log_success(f"Split into {len(texts)} chunks of text.")
    
    log_info("Ingesting chunks into Pinecone vector store...", Colors.BLUE)
    vectorstore.add_documents(texts)
    log_success("Ingestion complete!")

    #loader = TextLoader("./mediumblog.txt")
    #document = loader.load()
    #print(f"Loaded {len(document)} document(s)")
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #texts = text_splitter.split_documents(document)
    #print(f"Split into {len(texts)} chunks of text")
    #embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")

    #print ("Ingesting....")
    #vectorstore = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
