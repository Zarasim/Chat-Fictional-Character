import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_helper import (
    create_language_model,
    create_vectorstore,
    retrieve_vectorstore,
    create_retriever_chain,
    get_retrieval_chain_result,
)


st.set_page_config(page_title="Chat with a fictional character", page_icon="🤖")

st.title("Chat with a fictional character")

st.markdown(
    """
This is a chat application that allows you to communicate with a character.
You can send a message to the character and it will be displayed in the chat box.
    """
)

def set_new_chat_history(character):
    del st.session_state.chat_history
    st.session_state.chat_history = [
                AIMessage(content=f"I am an assistant that answers as if I were {character}."),
            ]
    st.write("***Start a new conversation!***")


def setup_db(character):
    st.session_state.click = st.session_state.click.fromkeys(st.session_state.click, False)
    st.session_state.click[character] = True

    if st.session_state.vector_db_state[character] == False:
        st.write(f"Fetch data about {character}")
        st.session_state.vector_store = create_vectorstore(character)
        st.session_state.vector_db_state[character] = True
        set_new_chat_history(character)
       
    else:
        st.write(f"Retrieve data about {character}")
        st.session_state.vector_store = retrieve_vectorstore(character)
        if st.session_state.history_aware[character] is False:
            set_new_chat_history(character)
        st.session_state.history_aware = st.session_state.history_aware.fromkeys(st.session_state.history_aware, False)
        st.session_state.history_aware[character] = True


chars_names = ["Sherlock Holmes","Harry Potter","Gandalf","Gollum","Paul Atreides","Yoda","Jack Sparrow"]
emojis = ["🕵️‍♂️","⚯ ͛","🧙‍♂️","👽 💍","🏜 🐛","◝(ᵔᵕᵔ)◜","🏴‍☠️🍾"]

st_char_names = [f"{char_name}  {emoji}" for char_name,emoji in zip(chars_names,emojis)]

if "click" not in st.session_state:
    st.session_state.click = {char:False for char in chars_names}
    st.session_state.vector_db_state = {char:False for char in chars_names}    
    st.session_state.history_aware = {char:False for char in chars_names}
    st.session_state.chat_history = []

with st.sidebar:
    character = st.radio("Choose a character",st_char_names,index=None)
    if character:
        idx = st_char_names.index(character)
        char_name = chars_names[idx]
        setup_db(char_name)
    else:
        st.info("Please select a character")
        

user_query = st.chat_input("Type your message here")

user_query_is_active: bool = user_query is not None and user_query != ""
character_is_clicked: bool = sum(st.session_state.click.values()) == 1

if user_query_is_active and character_is_clicked:
    llm = create_language_model()
    retriever_chain = create_retriever_chain(llm,st.session_state.vector_store)
    response = get_retrieval_chain_result(llm,retriever_chain,st.session_state.chat_history, user_query)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
else:
    st.info("Please select a character and send a message")
