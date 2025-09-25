import streamlit as st 
import os
from src.langgraph.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.page_title = self.config.get_page_title()
        self.llm_provider = self.config.get_llm_provider()
        self.usecase = self.config.get_usecase()
        self.gemini_model = self.config.get_gemini_model()

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.page_title, layout="wide")
        st.title(self.page_title)
        
    


