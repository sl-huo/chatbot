import os
import streamlit as st
import tempfile
# import getpass
# import openai
# from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
# from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler



################
### Define functions

# load user document file
def load_document(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        # load document files
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    # split the text into sentences
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts

# define retriver function
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # load text
    texts = load_document(uploaded_files)
    # create embeddings and store in vectordb
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # using OpenAIEmbeddings() instead of HuggingFaceEmbeddings()
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_documents(texts, embeddings)
    # create retriever
    retriever = vectordb.as_retriever(
        # search_type="mmr", 
        search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

## define callback handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

## define callback handler
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def main():

    st.title("Q&A Chatbot 🤖")
    st.write("This is a chatbot that uses the OpenAI API to answer questions from your document.")

    # using streamlit to ask user enter the API key
    with st.sidebar:
        openai_key = st.text_input("Enter your OpenAI API key", key="api_key_openai", type="password")
    
    # setting the API key as an environment variable
    os.environ['OPENAI_API_KEY'] = openai_key
    
    # check if API key is entered
    if not openai_key:
        st.info("Please enter your OpenAI API key in sidebar to continue.")
        st.stop()

    uploaded_files = st.sidebar.file_uploader(label="Upload your PDF file here", type=["pdf"], accept_multiple_files=True)

    if not uploaded_files:
        st.info("Please upload PDF document to continue.")
        st.stop()

    # setup retriever
    retriever = configure_retriever(uploaded_files)

    # setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, max_history=1)

    # setup llm model
    llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-3.5-turbo",
        streaming=True)

    ### setup chain
    # Q&A retrievalQA model setup
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     verbose=True)
    
    # Conversational retrieval chain setup
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=retriever, 
        memory=memory, 
        verbose=True)

    ### set inital message
    # if "messages" not in st.session_state:
    #     st.session_state["messages"] = [
    #         {"role": "assistant", "content": "Hi, I'm a chatbot who can assist you with questions about your document. How can I help you?"}
    #         ]

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("Hi, I'm a chatbot who can assist you with questions about your document. How can I help you?")


    # for msg in st.session_state.messages:
    #     st.chat_message(msg["role"]).write(msg["content"])

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_question := st.chat_input(placeholder="Enter your question here..."):
        # st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)

        # get the answer
        # answer = qa.run(user_question)
        # display the answer
        # st.write(answer)

        # get the answer and display it
        with st.chat_message("assistant"):
            ### use streamlit callback handler
            # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # answer = qa.run(st.session_state.messages, callbacks=[st_cb])
            # st.session_state.messages.append({"role": "assistant", "content": answer})
            # st.write(answer)
            ### use print retrieval handler
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa.run(user_question, callbacks=[retrieval_handler, stream_handler])
            # st.write(response)

if __name__ == "__main__":
    main()




