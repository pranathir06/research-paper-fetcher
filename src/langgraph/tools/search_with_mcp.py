
from typing_extensions import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages

from src.langgraph.tools.google_scholar_mcp import mcp_config
from src.langgraph.state.state import ResearcherState
from src.langgraph.LLMS.geminillm import GeminiLLM
from src.langgraph.prompts.prompts import research_agent_prompt_with_mcp, compress_research_system_prompt, compress_research_human_message



_client = None
model = GeminiLLM().get_llm()

def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client
##======Think Tool=====
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Always use this tool after search or major reasoning steps. 
    Reflection must be structured into 4 labeled sections:

    - Findings: What concrete information has been gathered so far?
    - Gaps: What crucial information is still missing?
    - Quality: Do I have sufficient diversity/validity (e.g., cross-validation)?
    - Decision: Should I continue searching, refine, or conclude?

    Each section must have at least one complete sentence.
    Optionally, include a confidence score (1–10) on sufficiency of current results.

    Args:
        reflection: Structured reflection on research progress, gaps, and next steps.

    Returns:
        A confirmation message that the reflection has been recorded.
    """
    return f"Reflection recorded: {reflection}"


# ===== AGENT NODES =====

async def llm_call(state: ResearcherState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    # Get available tools from MCP server
    client = get_mcp_client()
    mcp_tools = await client.get_tools()
    #tools = [*mcp_tools, think_tool]
    # Filter out the problematic advanced search tool, only keep keyword search and author info
    tools = [tool for tool in mcp_tools if tool.name in ["search_google_scholar_key_words", "get_author_info"]]

    # Initialize model with tool binding
    model_with_tools = model.bind_tools(tools)

    # Process user input with system prompt
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_with_mcp)] + state["researcher_messages"]
            )
        ]
    }

async def tool_node(state: ResearcherState):
    """Execute tool calls using MCP tools.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    # Debug: Check if we have messages
    if not state.get("researcher_messages"):
        raise ValueError("No researcher_messages found in state")
    
    last_message = state["researcher_messages"][-1]
    print(f"Debug: Last message type: {type(last_message)}")
    print(f"Debug: Last message content: {getattr(last_message, 'content', 'No content')}")
    print(f"Debug: Last message tool_calls: {getattr(last_message, 'tool_calls', 'No tool_calls attribute')}")
    
    tool_calls = getattr(last_message, 'tool_calls', None)
    if not tool_calls:
        raise ValueError("No tool calls found in the last message")

    async def execute_tools():
        """Execute all tool calls. MCP tools require async execution."""
        # Get fresh tool references from MCP server
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        # Filter out the problematic advanced search tool, only keep keyword search and author info
        tools = [tool for tool in mcp_tools if tool.name in ["search_google_scholar_key_words", "get_author_info"]]
        tools_by_name = {tool.name: tool for tool in tools}

        # Execute tool calls (sequentially for reliability)
        observations = []
        for tool_call in tool_calls:
            print(tool_call["name"])
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                # think_tool is sync, use regular invoke
                observation = tool.invoke(tool_call["args"])
            else:
                # MCP tools are async, use ainvoke
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    messages = await execute_tools()

    return {"researcher_messages": messages}

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for further processing or reporting.

    This function filters out think_tool calls and focuses on substantive
    file-based research content from MCP tools.
    """

    system_message = compress_research_system_prompt
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]

    response = model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue with tool execution or compress research.

    Determines whether to continue with tool execution or compress research
    based on whether the LLM made tool calls.
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Continue to tool execution if tools were called
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, compress research findings
    return "compress_research"