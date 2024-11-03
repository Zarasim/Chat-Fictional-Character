import os
from typing import List, Any
from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EnvironmentManager:
    """Manages environment variables and configuration."""
    
    @staticmethod
    def load_environment_variables() -> None:
        """Load environment variables from .env file."""
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")

    @staticmethod
    def get_api_key() -> str:
        """
        Get the OpenAI API key from environment variables.
        
        Returns:
            str: The API key.
            
        Raises:
            ValueError: If the API key is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        return api_key


class LangChainService:
    """Handles LangChain-related operations and model creation."""
    
    def __init__(self):
        """Initialize the LangChain service."""
        self.api_key = EnvironmentManager.get_api_key()

    def create_language_model(self, temperature: float = 0.2) -> ChatOpenAI:
        """
        Create a ChatOpenAI language model instance.
        
        Args:
            temperature (float): The temperature parameter for the model.
            
        Returns:
            ChatOpenAI: The initialized language model.
        """
        return ChatOpenAI(api_key=self.api_key, temperature=temperature)

    def create_vectorstore(self, character: str) -> FAISS:
        """
        Create a FAISS vector store for a character.
        
        Args:
            character (str): The character name to fetch data for.
            
        Returns:
            FAISS: The initialized vector store.
            
        Raises:
            Exception: If there's an error creating the vector store.
        """
        try:
            loader = WebBaseLoader(f"https://en.wikipedia.org/wiki/{character}")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter()
            split_docs = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings(api_key=self.api_key)
            return FAISS.from_documents(split_docs, embeddings)
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def create_history_aware_retriever(
        self,
        vectorstore: FAISS,
        compression_temperature: float = 0
    ) -> ContextualCompressionRetriever:
        """
        Create a history-aware retriever.
        
        Args:
            vectorstore (FAISS): The vector store to use.
            compression_temperature (float): Temperature for the compression model.
            
        Returns:
            ContextualCompressionRetriever: The initialized retriever.
        """
        # Create the base retriever
        base_retriever = vectorstore.as_retriever()
        
        # Create the compression model
        compression_llm = ChatOpenAI(
            api_key=self.api_key,
            temperature=compression_temperature
        )
        
        # Create the document compressor
        compressor = LLMChainExtractor.from_llm(compression_llm)
        
        # Create and return the contextual compression retriever
        return ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )

    def get_retrieval_chain_result(
        self,
        llm: ChatOpenAI,
        retriever: ContextualCompressionRetriever,
        chat_history: List[Any],
        user_query: str
    ) -> str:
        """
        Generate a response using the retrieval chain.
        
        Args:
            llm (ChatOpenAI): The language model.
            retriever (ContextualCompressionRetriever): The retriever instance.
            chat_history (List[Any]): The chat history.
            user_query (str): The user's query.
            
        Returns:
            str: The generated response.
        """
        try:
            # Create a prompt template that includes the context variable
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that takes on the personality of the character being queried. Use the following context to inform your responses while maintaining the character's voice and mannerisms.\n\nContext: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Create the document chain with the updated prompt
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )
            
            # Create the retrieval chain
            chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=document_chain
            )
            
            # Execute the chain
            result = chain.invoke({
                "chat_history": chat_history,
                "input": user_query
            })
            
            return result.get("answer", "I apologize, but I couldn't generate a response.")
            
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")


# Initialize the service
_service = LangChainService()

# Export the functions to maintain backward compatibility
create_language_model = _service.create_language_model
create_vectorstore = _service.create_vectorstore
create_history_aware_retriever = _service.create_history_aware_retriever
get_retrieval_chain_result = _service.get_retrieval_chain_result

# Load environment variables at module import
EnvironmentManager.load_environment_variables()