import os

from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import Any, Dict, Union

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding=embeddings
)
# Initialize chat model
model = init_chat_model("llama3.2", model_provider="ollama")


@tool(
    response_format="content_and_artifact",
    description="Retrieve relevant documentation chunks from the Pinecone index for a user query."
)
def retrieve_context(query: Union[str, Dict[str, Any]]):
    if isinstance(query, dict):
        query = query.get("query") or query.get("text") or ""
        if not query:
            query = " ".join(f"{k}:{v}" for k, v in query.items()) if isinstance(query, dict) else str(query)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    