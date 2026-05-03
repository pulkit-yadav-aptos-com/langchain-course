from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)

# k/fetch_k/lambda_mult go to the vector store, not to invoke().
# MMR reduces near-duplicate overview chunks from different URLs.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 28, "lambda_mult": 0.55},
)

model = init_chat_model(model="gpt-oss:latest", model_provider="ollama")

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a documentation assistant. Answer using ONLY the excerpts below.\n"
            "- Answer the user's question directly first; stay on the topic they asked about.\n"
            "- If the excerpts only mention the topic briefly, summarize that faithfully — do not substitute a generic LangChain product overview.\n"
            "- If the excerpts do not contain enough information, say you do not know.\n"
            "- Do not invent APIs, version-specific claims, or features not supported by the excerpts.",
        ),
        (
            "human",
            "Excerpts from the documentation (each block lists its source URL):\n\n"
            "{context}\n\nQuestion: {question}",
        ),
    ]
)


def _dedupe_documents(docs: List[Document], limit: int = 6) -> List[Document]:
    seen: set[str] = set()
    unique: List[Document] = []
    for doc in docs:
        key = f"{doc.metadata.get('source', '')}||{doc.page_content[:400]}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= limit:
            break
    return unique


def _format_context(docs: List[Document]) -> str:
    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "Unknown")
        parts.append(f"--- Excerpt {i} (source: {src}) ---\n{doc.page_content}")
    return "\n\n".join(parts)


def run_llm(query: str) -> Dict[str, Any]:
    """Retrieve relevant docs, then answer from those excerpts only."""
    retrieved = list(retriever.invoke(query))
    context_docs = _dedupe_documents(retrieved)
    context = _format_context(context_docs)
    chain = RAG_PROMPT | model
    response = chain.invoke({"question": query, "context": context})
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    while True:
        query = input("Enter a question: ")
        if query.lower() in ["exit", "quit", "bye"]:
            break
        result = run_llm(query)
        print(f"Answer: {result['answer']}")
        print(f"Context: {result['context']}")
        print("-" * 50)