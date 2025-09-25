from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
load_dotenv()

def get_tools():
    """
    Return the list of tools to be used in the chatbot
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tools=[TavilySearchResults(max_results=5)]
    return tools

def create_tool_node(tools):
    """
    creates and returns a tool node for the graph
    """
    return ToolNode(tools=tools)