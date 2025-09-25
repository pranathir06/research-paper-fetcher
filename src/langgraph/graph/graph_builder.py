from langgraph.graph import StateGraph,START,END
from src.langgraph.state.state import State,DeviceSearchState, ResearcherState,ResearcherOutputState
from src.langgraph.nodes.chatbot_with_tool_node import ChatbotWithToolNode
from src.langgraph.nodes.nodes import DeviceSearchNode
from src.langgraph.tools.search_tool import get_tools,create_tool_node
from src.langgraph.tools.search_with_mcp import llm_call, tool_node, compress_research,should_continue
#from IPython.display import Image, display

#from langgraph.prebuilt import tools_condition,ToolNode

class GraphBuilder:
    def __init__(self,model):
        self.llm=model
        #self.chain=chain
        #self.graph_builder=StateGraph(ResearcherState)

    
    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node
        tools=get_tools()
        prebuilt_tool_node=create_tool_node(tools)

        ## Define the LLM
        llm=self.llm
        #chain=self.chain

        ## Define the chatbot node
        """
        obj_chatbot_with_node=ChatbotWithToolNode(llm)
        chatbot_node=obj_chatbot_with_node.create_chatbot(tools)
        prompt_node=obj_chatbot_with_node.create_prompt_only(chain)
        ## Add nodes
        self.graph_builder.add_node("prompt",prompt_node)
        self.graph_builder.add_node("chatbot",chatbot_node)
        self.graph_builder.add_node("tools",prebuilt_tool_node)
        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"prompt")
        self.graph_builder.add_edge("prompt","chatbot")
        self.graph_builder.add_conditional_edges("chatbot",tools_condition)
        self.graph_builder.add_edge("tools","chatbot")
        self.graph_builder.add_edge("chatbot",END)

        return self.graph_builder.compile()
        

        obj_chatbot_with_node=DeviceSearchNode(llm)
        refine_query_node=obj_chatbot_with_node.create_refine_query()
        search_device_node=obj_chatbot_with_node.create_search_device()
        reflect_results_node=obj_chatbot_with_node.create_reflect_results()
        compress_results_node=obj_chatbot_with_node.create_compress_results()
        
        def conditional_edge(state: DeviceSearchState):
            if state.get("action")=="enough":
                return "compress_results"
            else:
                return "search_device"
            

        ## Add nodes
        self.graph_builder.add_node("refine_query",refine_query_node)
        self.graph_builder.add_node("search_device",search_device_node)
        self.graph_builder.add_node("reflect_results",reflect_results_node)
        self.graph_builder.add_node("compress_results",compress_results_node)

        ## Define conditional and direct edges
        self.graph_builder.add_edge(START,"refine_query")
        self.graph_builder.add_edge("refine_query","search_device")
        self.graph_builder.add_edge("search_device","reflect_results")
        self.graph_builder.add_conditional_edges("reflect_results",conditional_edge)
        #self.graph_builder.add_conditional_edges("reflect_results","compress_results",lambda x: x=="enough")
        self.graph_builder.add_edge("compress_results",END)

        researcher_agent=self.graph_builder.compile()
        
        

        # Show the agent
        #display(Image(researcher_agent.get_graph().draw_mermaid_png()))
        """
        agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

        # Add nodes to the graph
        agent_builder_mcp.add_node("llm_call", llm_call)
        agent_builder_mcp.add_node("tool_node", tool_node)
        agent_builder_mcp.add_node("compress_research", compress_research)

        # Add edges to connect nodes
        agent_builder_mcp.add_edge(START, "llm_call")
        agent_builder_mcp.add_conditional_edges(
            "llm_call",
            should_continue,
            {
                "tool_node": "tool_node",        # Continue to tool execution
                "compress_research": "compress_research",  # Compress research findings
            },
        )
        agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing
        agent_builder_mcp.add_edge("compress_research", END)
        #print("hiii")
        # Compile the agent
        agent_mcp = agent_builder_mcp.compile()
        

        return agent_mcp