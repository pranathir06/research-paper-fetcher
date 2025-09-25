'''
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json


class DisplayResultStreamlit:
    def __init__(self,graph,user_message):
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        graph = self.graph
        user_message = self.user_message
        print(user_message)
        
        # Prepare state and invoke the graph
        initial_state = {"messages": [user_message]}
        res = graph.invoke(initial_state)
        for message in res['messages']:
            if type(message) == HumanMessage:
                with st.chat_message("user"):
                    st.write(message.content)
            elif type(message)==ToolMessage:
                with st.chat_message("ai"):
                    st.write("Tool Call Start")
                    st.write(message.content)
                    st.write("Tool Call End")
            elif type(message)==AIMessage and message.content:
                with st.chat_message("assistant"):
                    st.write(message.content)

'''

import streamlit as st

class DisplayDeviceSearchStreamlit:
    def __init__(self, graph, device_name, manufacturer=""):
        self.graph = graph
        self.device_name = device_name
        self.manufacturer = manufacturer

    def display_result_on_ui(self):
        st.chat_message("user").write(f"{self.device_name} ({self.manufacturer})" if self.manufacturer else self.device_name)

        # Prepare initial state for LangGraph
        state = {
            "device": self.device_name,
            "manufacturer": self.manufacturer,
            "query": "",
            "raw_results": [],
            "retries": 0,
            "action": "",
            "relevant_urls": [],
            "final_report": ""
        }

        # Refine query first
        from copy import deepcopy
        #state = deepcopy(state)  # optional safety copy
        print(state)
        st.write(state)
        #state = self.graph.nodes["refine_query"](state)

        # Invoke the LangGraph pipeline
        result = self.graph.invoke(state)

        # Display raw results as Tool Calls (optional)
        if result["raw_results"]:
            with st.chat_message("ai"):
                st.write("🔹 Tavily Search Results (raw)")
                for r in result["raw_results"]:
                    st.write(r)

        # Display final cleaned report
        if result.get("final_report"):
            with st.chat_message("assistant"):
                st.write("✅ Final Device Research Report")
                st.write(result["relevant_urls"])
