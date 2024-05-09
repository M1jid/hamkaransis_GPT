
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
import streamlit as st


def get_pdf_text(pdf_docs='C:/Users/itel/Desktop/your work data.pdf'):
  pdf_reader = [PyPDFLoader(pdf_docs)]
  embeddings = HuggingFaceInstructEmbeddings(model_name="SajjadAyoubi/xlm-roberta-large-fa-qa")
  index  = VectorstoreIndexCreator( embedding=embeddings, text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)).from_loaders(pdf_reader)
  return index

OPENAI_API_KEY='your api key'
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='your api key',
    model="Qwen/Qwen1.5-110B-Chat",)

index = get_pdf_text()
chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type='stuff',
  retriever=index.vectorstore.as_retriever(), input_key='question')

st.title('به چت بات همکاران سیستم خوش امدید')
with st.spinner('درحال پردازش'):
    prompt = st.chat_input('چطور میتونم  کمک کنم؟')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        response = chain.run(f'persian{prompt}')
        helpful_answer = response.split("Helpful Answer:")[-1]
        st.chat_message('ai').markdown(helpful_answer)
        st.session_state.messages.append({'role': 'ai', 'content': helpful_answer})






