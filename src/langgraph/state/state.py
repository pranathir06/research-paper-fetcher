from pydantic import BaseModel, Field
from typing_extensions import TypedDict,Annotated,List,Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated,List,Dict
import operator

class State(TypedDict):
    """
    Represents the structure of the state used in graph.
    """
    messages: Annotated[list,add_messages]


class DeviceSearchState(TypedDict,total=False):
    device: str
    manufacturer: str
    query: str
    raw_results: List[str]
    retries: int
    action: str
    relevant_urls: List[str]
    final_report: str

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
