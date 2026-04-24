from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

#pydantic imports for structural outputs
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

tavily = TavilyClient()


class Source(BaseModel):
    """ schema for a source used by the agent """
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answers and sources"""
    answer: str = Field(description="Schema for agent response with answers and sources")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

@tool
def search(query: str) -> str:
    """
    A tool that searches over internet
    Args: 
        query: The query to search for
    Returns: 
        The search result
    """
    print("\n\n\n")
    print(f"Searching for {query}")
    print("\n\n\n")
    return tavily.search(query=query)

llm = ChatGroq(temperature=0, model="llama-3.1-8b-instant")
tools = [search]
agent = create_agent(llm, tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an AI engineer in the langchain on linked in and therir details")})
    print(result["structured_response"])


if __name__ == "__main__":
    main()
