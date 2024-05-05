import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import GPT4AllEmbeddings

import streamlit as st

# from watsonxlangchain import LangChainInterFace


# @st.cache_resource
def get_pdf_text(pdf_docs='C:/Users/itel/Desktop/Question.pdf'):
  pdf_reader = [PyPDFLoader(pdf_docs)]
  # text = ""

  # for page in pdf_reader.pages:
  #   text += page.extract_text()
  # text_splitter = CharacterTextSplitter(
  #   separator="\n",
  #   chunk_size=450,
  #   chunk_overlap=50,
  #   length_function=len
  # )
  # text = text_splitter.split_text(text)

  embeddings = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-small-v2")
  index  = VectorstoreIndexCreator( embedding=embeddings, text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)).from_loaders(pdf_reader)
  return index


access_token = "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"
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=access_token,
                     model_kwargs={"temperature": 0.5, "max_length":1000})

st.session_state.index =get_pdf_text()

chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type='stuff',
  retriever=st.session_state.index.vectorstore.as_retriever(), input_key='question')

st.title('به چت بات همکاران سیستم خوش امدید')
with st.spinner('درحال پردازش'):
  if 'messages' not in st.session_state:
    st.session_state.messages = []
  for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

  prompt = st.chat_input('چطور میتونم  کمک کنم؟')



  if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})


    response = chain.run(prompt)
    st.chat_message('assistent').markdown(response)
    st.session_state.messages.append(
      {'role':'assistent','content':response}
    )



