import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from typing_extensions import Literal

from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiLLM:
    def __init__(self):
        self.model = "gemini-2.5-pro"
        self.temperature = 0
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.llm = init_chat_model(model="google_genai:gemini-2.5-pro",temperature=0.0,client=None,       # optional
                                           api_key=self.api_key    # ⚠️ explicit API key
                                            )

        """
        system_prompt='''You are a research agent.  

                        Your only job is to find research papers using the Tavily tool.  
                        Follow these strict rules:

                        1. Always attempt to return **5 research papers** most relevant to the user query.  
                        2. If you cannot find 5 papers on the first try, refine your search and retry (up to 3 total search loops).  
                        3. If after 3 attempts you still do not have 5 papers, return the number of papers you found (even if fewer than 5).  
                        4. For each paper, include exactly:
                        - The paper title  
                        - A direct clickable URL (prefer PDF such as arXiv or publisher DOI page).  
                        5. If multiple URLs exist, pick the most direct PDF or publisher page.  
                        6. Do not include abstracts, authors, metadata, explanations, or extra text.  
                        7. Output format must be a clean numbered list like this:
                        <title> — <url>
                        <title> — <url>
                        <title> — <url>
                        <title> — <url>
                        <title> — <url>


                        8. If fewer than 5 results are found after 3 retries, return the available results in the same numbered list format.  
                        9. Never add commentary before or after the list. Only output the results.

                        IMPORTANT:
                        1. The user query is always about medical devices. 
                        2. The devices are specifically related to opthalmology.
                        3. when you get response from the tool,check whether it is relevant to the user query or not. If not, try to search again with a different query. 
                        4. go deep into the search and find the most relevant research papers.
                        5. Give the links for web pages or research papers or anything that is relevant to the user query.
                        '''
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                #MessagesPlaceholder("messages"),
                ("human", "{question}"),
            ]
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        """

    def get_llm(self):
        return self.llm
    #def get_chain(self):
        #return self.llm_chain
