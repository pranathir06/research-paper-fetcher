from src.langgraph.state.state import DeviceSearchState
from src.langgraph.LLMS.geminillm import GeminiLLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import ChatPromptTemplate
from src.langgraph.prompts.prompts import device_search_system_prompt, compress_device_results_prompt, compress_device_simple_human_message,device_relevance_system_prompt
import json


tavily = TavilySearchResults(k=10)
llm = GeminiLLM().get_llm()

class DeviceSearchNode:
    """
    Node to handle device search logic.
    """

    def __init__(self,model):
            self.llm = model

    # ========== Node 1: Refine Query ==========
    def create_refine_query(self):

        def refine_query(state: DeviceSearchState):
            print(state)
            manufacturer = state.get("manufacturer", "").strip()
            if manufacturer:
                refined_query = f"{state["device"]} ophthalmology medical device {manufacturer}"
            else:
                refined_query = f"{state["device"]} ophthalmology medical device"
            
            state["query"] = refined_query
            return state
        return refine_query
    

    # ========== Node 2: Search ==========
    def create_search_device(self):
        def search_device(state: DeviceSearchState):
            results = tavily.invoke({"query": state["query"]})
            tavily_tool = TavilySearchResults(max_results=5, topic="medical_devices", llm=llm)
            llm
            urls = [f"{r['title']} : {r['url']}" for r in results]
            state["raw_results"].extend(urls)
            return state
        return search_device
    

    # ========== Node 3: Reflection ==========
    def create_reflect_results(self):
        def reflect_results(state: DeviceSearchState):
            # Prepare the last 10 search results
            results_text = "\n".join(state["raw_results"][-10:])
            prompt = device_relevance_system_prompt(
                user_query=state["query"],
                manufacturer=state.get("manufacturer", ""),
                search_results="\n".join(state["raw_results"][-10:])
            )

            # Ask LLM to evaluate relevance
            response = llm.invoke(prompt)
            print(response.content)

            try:
                relevance_scores = json.loads(response.content)
                # Make sure all items are dicts with 'result' and 'relevant'
                relevance_scores = [
                    r if isinstance(r, dict) and "result" in r and "relevant" in r
                    else {"result": r, "relevant": True}
                    for r in relevance_scores
                ]
            except json.JSONDecodeError:
                # fallback: mark all last 10 raw results as relevant
                print("JSON decode error, marking all last 10 results as relevant")
                relevance_scores = [{"result": r, "relevant": True} for r in state["raw_results"][-10:]]
                    
            

            # Add URLs marked as relevant to the set
            for r in relevance_scores:
                if r.get("relevant"):
                    # extract URL from result string
                    if ":" in r["result"]:
                        url = r["result"].split(":")[-1].strip()
                        if url not in state["relevant_urls"]:
                            state["relevant_urls"].append(url)

            # Decide next action
            if len(state["relevant_urls"]) >= 3:
                state["action"] = "enough"
            elif state["retries"] < 3:
                state["retries"] += 1
                state["action"] = "retry"
            else:
                state["action"] = "enough"

            return state

        return reflect_results


    # ========== Node 4: Compress Results ==========
    def create_compress_results(self):

        def compress_results(state: DeviceSearchState):
            compression_prompt = ChatPromptTemplate.from_messages([
                ("system", compress_device_results_prompt),
                ("user", "\n".join(state["raw_results"])),
                ("human", compress_device_simple_human_message),
            ])
            report = llm.invoke(compression_prompt.format())
            state["final_report"] = report.content
            return state
        return compress_results