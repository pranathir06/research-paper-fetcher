from src.langgraph.state.state import State


class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """
    def __init__(self,model):
        self.llm = model
    


    def create_chatbot(self, tools):
        """
        Returns a chatbot node function.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Chatbot logic for processing the input state and returning a response.
            """
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        return chatbot_node
    
    def create_prompt_only(self, llm_chain):
        """
        Returns a node function that uses only prompt/LLM chain.
        """
        def prompt_only_node(state: State):
            user_msg = state["messages"][-1]  # last message from user
            response = llm_chain.invoke({"question": user_msg.content})
            return {"messages": state["messages"] + [response["text"]]}

        return prompt_only_node
