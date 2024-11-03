import streamlit as st
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_helper import (
    create_language_model,
    create_vectorstore,
    create_history_aware_retriever,
    get_retrieval_chain_result,
)


class ChatApp:
    """Main chat application class handling the Streamlit interface and chat logic."""
    
    CHARACTERS = [
        "Sherlock Holmes",
        "Harry Potter",
        "Gandalf",
        "Gollum",
        "Paul Atreides",
        "Yoda",
        "Jack Sparrow",
    ]

    def __init__(self):
        """Initialize the chat application."""
        self.setup_page_config()
        self.llm: Optional[BaseChatModel] = None
        self.vectorstore: Optional[VectorStore] = None

    @staticmethod
    def setup_page_config() -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="🤖 Chat with a Fictional Character",
            page_icon="🤖",
            layout="wide"
        )
        st.title("🤖 Chat with a Fictional Character")
        st.markdown(
            """
            This chat application allows you to communicate with a character.
            Send a message to the character and it will be displayed in the chat box.
            """
        )

    def initialize_session(self, character: str) -> None:
        """
        Initialize or reset the chat session for a new character.
        
        Args:
            character (str): The name of the character to interact with.
        """
        st.session_state.chat_history = [
            AIMessage(content=f"I am an assistant that answers as if I were {character}.")
        ]
        st.write("***Start a new conversation!***")

    def setup_vectorstore(self, character: str) -> None:
        """
        Initialize the vector store for the selected character.
        
        Args:
            character (str): The name of the character to fetch data for.
        """
        try:
            self.vectorstore = create_vectorstore(character)
            st.session_state.vectorstore = self.vectorstore
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
            return None

    def handle_user_input(self, user_input: str) -> None:
        """
        Process user input and generate a response.
        
        Args:
            user_input (str): The user's message.
        """
        try:
            if not self.llm:
                self.llm = create_language_model()
            
            retriever = create_history_aware_retriever(st.session_state.vectorstore)
            response = get_retrieval_chain_result(
                self.llm,
                retriever,
                st.session_state.chat_history,
                user_input
            )
            
            # Update chat history
            st.session_state.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response)
            ])
            
            # Display response
            st.write(f"🤖 {response}")
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

    def render_chat_interface(self) -> None:
        """Render the chat interface components."""
        character = st.selectbox("Choose a character:", self.CHARACTERS)
        
        if st.button("✨ Start New Conversation ✨"):
            self.initialize_session(character)
            self.setup_vectorstore(character)

        if "chat_history" in st.session_state and "vectorstore" in st.session_state:
            user_input = st.text_input("🗨️ You:", key="user_input")
            
            if st.button("➡️ Send") and user_input:
                self.handle_user_input(user_input)

    def run(self) -> None:
        """Run the chat application."""
        self.render_chat_interface()


def main() -> None:
    """Main function to initialize and run the chat application."""
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()