import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="gpt-oss:latest")

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    f"""Answer the question based only on the following context:
    {{context}}
    Question: {{question}}
    Provide a detailed answer:
    """
)


def format_docs(docs):
    """Format the documents to be used as context for the question"""
    return "\n\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_lcel(query:str):

    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(messages)
    return response.content

def retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Starting the chat...")
    while True:
        question = input("Enter a question: ")
        if question.lower() == "exit":
            break
        response = retrieval_chain_with_lcel().invoke({"question": question})
        print(response)