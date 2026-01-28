import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

chat_model = init_chat_model(
    model="nomic-llama2-70b-chat",
    model_provider="ollama",
    temperature=0.2,
)

@tool
def retrieve_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}"
        for d in docs
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that answers questions about LangChain documentation. "
            "You have access to a tool that retrieves relevant documentation. "
            "Use the tool before answering. "
            "Always cite sources by repeating their Source fields. "
            "If the context does not contain the answer, say \"I don't know.\"",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(chat_model, [retrieve_context], prompt)
executor = AgentExecutor(agent=agent, tools=[retrieve_context], verbose=True)

def run_llm(query: str) -> Dict[str, Any]:
    docs = retriever.invoke(query)
    res = executor.invoke({"input": query})
    return {"answer": res["output"], "context_documents": docs}

if __name__ == "__main__":
    query = "What are deep agents?"
    result = run_llm(query)
    print("Answer:", result["answer"])
    print("Context Documents:", result["context_documents"])