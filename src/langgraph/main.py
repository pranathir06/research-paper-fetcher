"""
import streamlit as st
from src.langgraph.ui.streamlit.loadui import LoadStreamlitUI
from src.langgraph.LLMS.geminillm import GeminiLLM
from src.langgraph.graph.graph_builder import GraphBuilder 
from src.langgraph.ui.streamlit.display_result import DisplayDeviceSearchStreamlit

def load_langgraph_app():
   

    ##Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    #user_message = st.text_input("Enter the device name here...")
    #manufacturer_name = st.text_input("Enter the manufacturer name here (optional)...")
    user_query = st.text_area("Enter your research question:", height=150)

    if user_query:
        #try:
            ## Configure LLM model
            model = GeminiLLM().get_llm()
            #chain=GeminiLLM().get_chain()

            graph_builder = GraphBuilder(model)
            #try:
            graph=graph_builder.chatbot_with_tools_build_graph()
            DisplayDeviceSearchStreamlit(graph,user_message,manufacturer_name).display_result_on_ui()
            #except Exception as e:
                #st.error(f"Error building graph: {e}")
                #return

        #except Exception as e:
         #   st.error(f"Error configuring LLM model: {e}")
          #  return
"""


import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage
from src.langgraph.state.state import ResearcherState
from src.langgraph.LLMS.geminillm import GeminiLLM
from src.langgraph.graph.graph_builder import GraphBuilder 
#from src.langgraph.ui.streamlit.display_result import DisplayDeviceSearchStreamlit

def load_langgraph_app():
    """
    Loads and runs the Langgraph Deep Research Agent with Streamlit UI.
    This skips the scoping phase and directly runs research with GraphBuilder, GeminiLLM,
    and DisplayDeviceSearchStreamlit.
    """

    st.set_page_config(page_title="Deep Research Agent", layout="wide")

    st.title("🔍 Deep Research Agent")
    st.write("Ask a research question and let the agent research it for you.")

    # Input box
    user_query = st.text_area("Enter your research question:", height=150)

    if st.button("Run Research"):
        if not user_query.strip():
            st.warning("Please enter a research question.")
        else:
            #try:
                # Configure LLM
            model = GeminiLLM().get_llm()

                # Build graph
            graph_builder = GraphBuilder(model)
            graph = graph_builder.chatbot_with_tools_build_graph()

                # Phase: Direct Research Agent
            st.subheader("📌 Research Agent Findings")
            research_state = ResearcherState(
                researcher_messages=[HumanMessage(content=user_query)]
            )

                # Run the async research graph
            research_output = asyncio.run(graph.ainvoke(research_state))

                # Display compressed summary
            st.success("📑 Compressed Research Summary:")
            st.write(research_output)

                # Show raw notes if present
            if "raw_notes" in research_output:
                with st.expander("🔎 Raw Notes from Research"):
                    for note in research_output["raw_notes"]:
                        st.write(note)

                # Device-specific UI display
                """DisplayDeviceSearchStreamlit(
                    graph, user_query, None  # manufacturer_name optional
                ).display_result_on_ui()"""

            #st.write("---END OF THE DISCUSSION---")

            #except Exception as e:
             #   st.error(f"Error running research: {e}")
