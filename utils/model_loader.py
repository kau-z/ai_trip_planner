import os
from dotenv import load_dotenv
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class configLoader:
    def __init__(self):
        print("Loading config....")
        self.config = load_config()

    def __getitem__(self, key):
        return self.config.get(key)
    

class ModelLoader(BaseModel):
    model_provider: Literal["groq", "openai"] = "groq"
    config: Optional[configLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = configLoader()

    class Config:
        arbitary_types_allowed = True

    def load_llm(self):
        """
        Load and return the LLM model.
        """
        print("LLM loading...")
        print(f"Loading LLM model from provider: {self.model_provider}")
        if self.model_provider == "openai":
            print("Loading OpenAI model...")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            model_name = self.config["llm"]["openai"]["model_name"]
            llm=ChatOpenAI(model_name="o4-mini", api_key=openai_api_key)
        elif self.model_provider == "groq":
            print("Loading Groq model...")
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_name = self.config["llm"]["groq"]["model_name"]
            llm=ChatGroq(model_name="deepseek-r1-distill-llama-70b", api_key=groq_api_key)

        return llm