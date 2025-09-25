from configparser import ConfigParser



class Config:
    def __init__(self, config_file: str = "src/langgraph/ui/uiconfigfile.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_page_title(self) -> str:
        return self.config.get("DEFAULT", "PAGE_TITLE", fallback="LangGraph : Build AI Applications with Graphs")

    def get_llm_provider(self) -> str:
        return self.config.get("DEFAULT", "LLM_PROVIDER", fallback="Gemini")

    def get_usecase(self) -> str:
        return self.config.get("DEFAULT", "USECASE", fallback="Research Assistant")

    def get_gemini_model(self) -> str:
        return self.config.get("DEFAULT", "GEMINI_MODEL", fallback="gemini-2.5-pro")