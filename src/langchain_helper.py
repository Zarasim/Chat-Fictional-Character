## main function used as backend of the appplication
# This function calls the OpenAI api, the FAISS vector store, and create the chatbotmessage object
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever


load_dotenv()

## Define large language model object
def create_language_model():
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)
    return llm

## Fetch data from wikipedia pages of characters
def create_vectorstore(character):

    # You can also load multiple webpages at once by passing in a list of urls to the loader. 
    # This will return a list of documents in the same order as the urls passed in.
    loader = WebBaseLoader(f"https://en.wikipedia.org/wiki/{character}")
    docs = loader.load()

    """
    This text splitter is the recommended one for generic text. 
    It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. 
    The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, 
    as those would generically seem to be the strongest semantically related pieces of text.
    """
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    # create embeddings
    embeddings = OpenAIEmbeddings()

    # create vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(f"./src/faiss_index_{character}")

    return vectorstore

def retrieve_vectorstore(character):
    # create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(f"./src/faiss_index_{character}", embeddings,allow_dangerous_deserialization = True)
    return vectorstore

def create_retriever_chain(llm,vectorstore):
    # First we need a prompt that we can pass into an LLM to generate this search query
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    # If there is no chat_history, then the input is just passed directly to the retriever. If there is chat_history, 
    # then the prompt and LLM will be used to generate a search query. That search query is then passed to the retriever.
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_retrieval_chain_result(llm,retriever_chain,chat_history,user_query):

    prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions as if you were the character described from the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ])

    # Create a chain for passing a list of Documents to a model as {context}.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Combine documen_chain and retriever to generate a search query based on the docstore
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    
    result = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": user_query})

    return result["answer"]




