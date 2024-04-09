import streamlit as st


def get_db(character):
    st.session_state.click = st.session_state.click.fromkeys(st.session_state.click, False)
    st.session_state.click[character] = True
    st.write(f"status db character: {st.session_state.vector_db_state[character]}")
    
    if st.session_state.vector_db_state[character] == False:
        st.write(f"create db for {character}")
        st.session_state.vector_db_state[character] = True
        st.write(f"status db character: {st.session_state.vector_db_state[character]}")
    else:
        st.write(f"retrieve db for {character}")


st.title("Chat with a fictional character")

st.markdown(
    """
This is a chat application that allows you to communicate with a fictional character.
    """
)


chars_names = ['Sherlock 🕵️‍♂️','Harry Potter 🧙‍♂️','Gandalf']

if "click" not in st.session_state:
    st.session_state.click = {char:False for char in chars_names}
    st.session_state.vector_db_state = {char:False for char in chars_names}

character = st.radio("characters",chars_names,index=None)

if character:
    get_db(character)

st.write(f"The variable character return {character}")


user_query = st.chat_input("Type your message here")

user_query_is_active: bool = user_query is not None and user_query != ""
character_is_clicked: bool = sum(st.session_state.click.values()) == 1

# SIMPLE STEP 1: Always create vectordatabase after selecting a character 
if user_query_is_active and character_is_clicked:
    st.info("The query will be processed")
else:
    st.info("Please select a character")