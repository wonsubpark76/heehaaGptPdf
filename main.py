__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# buy me a coffee
button(username="heehaa", floating=True, width=221)

# Title
st.title("Chat PDFs")
st.subheader("PDFファイルをアップロードしてChatで質問する！", divider='rainbow')
st.write("")

# OpenAI KEY入力してもらう
openai_key = st.text_input('あなたのOpenAI API KEYを入力してください。', type="password")
st.write("<a href='https://teamladybird.com/%e3%81%9d%e3%81%ae%e4%bb%96%e6%8a%80%e8%a1%93/open-ai%e3%81%aeapi-key%e3%82%92%e4%bd%9c%e3%81%a3%e3%81%a6%e3%81%bf%e3%82%88%e3%81%86/' target='_blank'>OpenAIのAPIキーの取得方法はこちらへ</a>",unsafe_allow_html=True)
st.write("")

# 複数のPDFをもらう
def pdf_to_document(uploaded_files):
    page=[]
    for uploaded_file in uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        page.append(loader.load_and_split())
    return page

if openai_key:
    uploaded_files = st.file_uploader("PDFファイルアップロードしてください。",accept_multiple_files=True,type=['pdf'])
    st.write("---")
    # uploadしたら動く
    if len(uploaded_files) > 0:
        pages = pdf_to_document(uploaded_files)

        # Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap  = 20,
            length_function = len,
            is_separator_regex = False,
        )
        st.write(pages)
        texts = []
        for p in pages:
            texts.append(text_splitter.split_documents(p))
        #texts = text_splitter.split_documents(pages)

        # Embedding
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

        # load it into Chroma
        # from_documentsのパラメータを追加するとDBから取得することも可能
        db = Chroma.from_documents(texts, embeddings_model)

        # Stream をもらう Hander 作成
        from langchain.callbacks.base import BaseCallbackHandler
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text=initial_text
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text+=token
                self.container.markdown(self.text)

        # Question
        st.header("PDFへ質問してみてください。")
        question = st.text_input('質問を入力して「質問する」ボタンを押下')

        if st.button('質問する'):
            with st.spinner('答えを探してます。しばらくお待ちください。'):
                chat_box = st.empty()
                stream_hander = StreamHandler(chat_box)
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
                qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
                qa_chain({"query": question})