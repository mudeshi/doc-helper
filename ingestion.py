import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50, max_concurrency: int = 2):
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")

    sem = asyncio.Semaphore(max_concurrency)

    async def add_batch(batch: List[Document], batch_num: int, attempts: int = 5):
        async with sem:
            for t in range(attempts):
                try:
                    await vectorstore.aadd_documents(batch)
                    log_success(
                        f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                    )
                    return True
                except Exception as e:
                    msg = str(e)
                    if "Session is closed" in msg and t < attempts - 1:
                        await asyncio.sleep(2 * (2 ** t))
                        continue
                    log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
                    return False

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    successful = sum(1 for r in results if r is True)

    if successful == len(batches):
        log_success(f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully")


async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("🗺️  TavilyCrawl: Starting to crawl the documentation site", Colors.PURPLE)

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    all_docs: List[Document] = []
    for item in res["results"]:
        url = item.get("url")
        raw = item.get("raw_content") or ""
        log_info(f"TavilyCrawl: Successfully crawled {url} from documentation site")
        all_docs.append(Document(page_content=raw, metadata={"source": url}))

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 1200 chunk size and 150 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    splitted_docs = text_splitter.split_documents(all_docs)

    MAX_CHARS = 6000
    splitted_docs = [
        Document(page_content=d.page_content[:MAX_CHARS], metadata=d.metadata) for d in splitted_docs
    ]

    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    await index_documents_async(splitted_docs, batch_size=50, max_concurrency=2)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
