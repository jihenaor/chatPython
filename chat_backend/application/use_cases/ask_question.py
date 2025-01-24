from adapters.secondary.langchain.chat_adapter_factory import ChatAdapterFactory

from typing import Dict, Optional
import os
import getpass

from langchain_core.messages import AIMessage, trim_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


# Load PDF documents as knowledge base

def load_documents():
    file_path = "../doc2.pdf"
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist or is not a valid file.")
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.load():  # Loading synchronously to avoid issues with event loops
        pages.append(page)
    return pages


# Define the graph workflow for the conversation
workflow = StateGraph(state_schema=MessagesState)


# Define a class to store user context
class UserContext:
    def __init__(self):
        self.context = {}

    def update_context(self, session_id: str, key: str, value: str):
        if session_id not in self.context:
            self.context[session_id] = {}
        self.context[session_id][key] = value

    def get_context(self, session_id: str, key: str):
        return self.context.get(session_id, {}).get(key, None)


# Create a global user context object
user_context = UserContext()

# Define a global vector store variable
vector_store = None


# Define the function that calls the model
def call_model(state: MessagesState, use_trim: bool = False, max_tokens: int = 2048):
    # Decide if we need to trim the messages
    if use_trim:
        trimmed_messages = trim_messages(
            messages=state["messages"],
            max_tokens=max_tokens,  # Ajusta según el límite de tokens del modelo
            token_counter=lambda messages: sum(len(message.content.split()) for message in messages),
            start_on="human",
            strategy="last",
            include_system=True
        )
    else:
        trimmed_messages = state["messages"]

    # Get adapter for the model
    model_name = state.get("model", "ollama")
    chat_adapter = ChatAdapterFactory().get_adapter(model_name)
    # Invoke the model with the trimmed or full messages
    response = chat_adapter.invoke(trimmed_messages)
    return {"messages": trimmed_messages + [AIMessage(content=response.content)]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory persistence to the workflow
memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)


# Modify use case to interact with different LLMs
class AskQuestionUseCase:
    def execute(self, session_id: str, query: str, model: str, additional_params: Optional[Dict[str, str]] = None,
                use_trim: bool = False, max_tokens: int = 2048) -> str:
        self._update_user_context(session_id, query)
        user_name = user_context.get_context(session_id, "name")
        template_messages, use_rag = self._prepare_template_messages(query, user_name, additional_params)

        if use_rag:
            # Cargar documentos y crear el vector store si no existe
            global vector_store
            if vector_store is None:
                try:
                    pages = load_documents()
                    if model == "ollama":
                        vector_store = InMemoryVectorStore.from_documents(pages, OllamaEmbeddings(model="llama3.2"))
                    else:
                        vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
                    print(f"Loaded {len(pages)} pages from knowledge base.")
                except FileNotFoundError as e:
                    print(e)
                    return "Error: Knowledge base file not found."

            # Recuperar documentos relevantes para la consulta sobre información de la empresa
            retriever = vector_store.as_retriever()
            retrieved_docs = retriever.invoke(query, k=2)
            if not retrieved_docs:
                print("No se encontraron documentos relevantes para la consulta.")
                return "No relevant documents found."
            retrieved_context = "".join(doc.page_content for doc in retrieved_docs)
            # Incluir el contexto en el mensaje del sistema
            system_prompt = f"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {retrieved_context}"""
            print(f"System Prompt con contexto recuperado: {system_prompt}")
            template_messages.insert(0, ("system", system_prompt))

        prompt_value = self._create_prompt(template_messages, query)
        return self._invoke_langgraph_app(session_id, prompt_value, model, use_trim, max_tokens)

    def _update_user_context(self, session_id: str, query: str):
        if "soy" in query.lower():
            name = query.split("soy")[-1].strip().split(",")[0].strip()
            user_context.update_context(session_id, "name", name)

    def _prepare_template_messages(self, query: str, user_name: Optional[str],
                                   additional_params: Optional[Dict[str, str]]) -> tuple:
        use_rag = False
        if additional_params and "query_type" in additional_params:
            query_type = additional_params["query_type"].lower()
            if query_type == "company_info":
                use_rag = True
                template_messages = [
                    ("human", f"Hello {user_name}!" if user_name else query),
                    ("human", query)
                ]
            elif query_type == "translation":
                input_language = additional_params.get("input_language", "English")
                output_language = additional_params.get("output_language", "Spanish")
                template_messages = [
                    ("system", f"You are a helpful assistant that translates {input_language} to {output_language}."),
                    ("human", query)
                ]
            else:
                template_messages = [
                    ("system", "You are a helpful assistant."),
                    ("human", query)
                ]
        else:
            template_messages = [
                ("system", "You are a helpful assistant."),
                ("human", f"Hello {user_name}!" if user_name else query)
            ]

        return template_messages, use_rag

    def _create_prompt(self, template_messages: list, query: str) -> ChatPromptTemplate:
        prompt_template = ChatPromptTemplate(template_messages)
        return prompt_template.invoke({"user_input": query})

    def _invoke_langgraph_app(self, session_id: str, prompt_value: ChatPromptTemplate, model: str, use_trim: bool,
                              max_tokens: int) -> str:
        input_messages = {"messages": prompt_value.messages, "model": model}
        config = {"configurable": {"thread_id": session_id}}
        state = MessagesState(**input_messages)
        output = call_model(state, use_trim, max_tokens)
        return output["messages"][-1].content
