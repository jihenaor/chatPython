from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define adapters for different LLMs
class ChatAdapterFactory:
    def __init__(self):
        print("Initializing ChatAdapterFactory...")
        
        # Initialize OpenAI first
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY no está configurado.")
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        # Initialize Ollama adapter
        print("Attempting to initialize Ollama adapter...")
        try:
            self.ollama_adapter = ChatOllama(
                model="llama3.2",
                temperature=0,
                verbose=True
            )
            print("✅ Ollama adapter initialized successfully with model: llama3.2")
        except Exception as e:
            print(f"❌ Error initializing Ollama adapter: {str(e)}")
            raise

        # Initialize adapters dictionary
        print("Setting up adapters dictionary...")
        self.adapters = {
            "ollama": self.ollama_adapter,
            "openai": ChatOpenAI(
                model="gpt-4",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key
            )
        }
        print("✅ ChatAdapterFactory initialization complete")

    def get_adapter(self, model_name: str):
        print(f"Requesting adapter for model: {model_name}")
        adapter = self.adapters.get(model_name)
        if adapter is None:
            print(f"⚠️ Warning: Model {model_name} not found, using default ollama")
            adapter = self.adapters["ollama"]
        return adapter