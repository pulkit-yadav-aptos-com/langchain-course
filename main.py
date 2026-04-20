from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


load_dotenv()

class Source(BaseModel):
    """Schema to be used by the agent"""
    url: str = Field(description="The url of the source of the information")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The answer to the user's question")
    sources: List[Source] = Field(default_factory=list, description="The sources used to answer the question")

llm = ChatOllama(
    model="gpt-oss:latest",
    temperature=0,
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({
    "messages": [
        HumanMessage(content="You must use the tools before answering: Give me a list of 10 job opening in the field of Agentic AI with langchain")
        ]
    })
    print(response)

if __name__ == "__main__":
    main()
