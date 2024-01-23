import streamlit as st
import time
import os
import pandas as pd


os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

from model.character.character import Character
from model.search.search import Search, Embedding
from model.search_sanghoon.search import Search as Search_sanghoon
from model.search_sunwoo.search import Search as Search_sunwoo

def main():
    selected_page = st.sidebar.selectbox("Select a page", [ "웹툰 검색하기 - 김선우", "웹툰 검색하기 - 한상훈", "캐릭터와 대화하기"])
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
    
    # if selected_page == "웹툰 검색하기 v1":
    #     search_page()
    if selected_page == "웹툰 검색하기 - 김선우":
        search_page_sunwoo()
    elif selected_page == "웹툰 검색하기 - 한상훈":
        search_page_sanghoon()
    elif selected_page == "캐릭터와 대화하기":
        character_page()

def search_page():
    st.title("이 웹툰 뭐였지 Demo Page")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=True)

    if uploaded_file is not None and len(st.session_state.search_messages) == 1:
        df = pd.read_csv(uploaded_file)
        embedding = Embedding(df)
        embedding_data = embedding.do_embedding()
        search = Search(embedding_data)
    elif uploaded_file is not None and len(st.session_state.search_messages) != 1:
        df = pd.read_csv(uploaded_file)
        embedding = Embedding(df)
        embedding_data = embedding.load_data()
        search = Search(embedding_data)
        
    st.subheader("웹툰을 검색해보세요!")
    st.caption("ex) 주인공이 못생겼다가 예뻐진 웹툰이 뭐지? \n 주인공이 힘순찐인 웹툰 알려줘 \n 수호라는 캐릭터가 등장하는 웹툰 알려줘 \n 무림 웹툰 추천해줘")
    
    if "search_messages" not in st.session_state:
        st.session_state["search_messages"] = [{"role": "assistant", "content": "안녕하세요. 웹툰에 대한 질문을 해주세요!"}]

    for msg in st.session_state.search_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input_key = "search_chat_input"
    # 사용자 인풋 받기  
    if prompt := st.chat_input("웹툰을 검색해보세요", key=chat_input_key):
        # 사용자 입력 보여 주기
        st.session_state.search_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 대화 보여 주기
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = search.receive_chat(prompt)
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.search_messages.append({"role": "assistant", "content": assistant_response})
        
def search_page_sunwoo():
    st.title("이 웹툰 뭐였지 Demo Page")
    
    uploaded_files = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=True)
    
    search_sunwoo = Search_sunwoo()
    pipeline_compression_retriever = None
        
    st.subheader("웹툰을 검색해보세요!")
    st.caption("ex) 주인공이 못생겼다가 예뻐진 웹툰이 뭐지? \n 주인공이 힘순찐인 웹툰 알려줘 \n 수호라는 캐릭터가 등장하는 웹툰 알려줘 \n 무림 웹툰 추천해줘")
    
    if "search_messages" not in st.session_state:
        st.session_state["search_messages"] = [{"role": "assistant", "content": "안녕하세요. 웹툰에 대한 질문을 해주세요!"}]

    for msg in st.session_state.search_messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if uploaded_files is not None and len(st.session_state.search_messages) > 0:
        datas = []
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            datas.append(df)
        if len(datas) > 0:
            # print("TEST!!!!!!!" + str(len(datas)))
            pipeline_compression_retriever = search_sunwoo.make_retriever(datas)

    chat_input_key = "search_chat_input_sunwoo"
    # 사용자 인풋 받기  
    if prompt := st.chat_input("웹툰을 검색해보세요", key=chat_input_key):
        # 사용자 입력 보여 주기
        st.session_state.search_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 대화 보여 주기
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # assistant_response = search.receive_chat(prompt)
            assistant_response = search_sunwoo.run(prompt, pipeline_compression_retriever)
            print(assistant_response)
            # for chunk in assistant_response.split():
            content = ""
            if len(assistant_response) > 0:
                for chunk in assistant_response:
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                    content = full_response
                message_placeholder.markdown(full_response)
            else:
                content = "검색 결과가 없습니다."
                message_placeholder.markdown("검색 결과가 없습니다.")
            # message_placeholder.markdown(assistant_response)
        st.session_state.search_messages.append({"role": "assistant", "content": content})
    
        
def search_page_sanghoon():
    st.title("이 웹툰 뭐였지 Demo Page")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=True)

    if uploaded_file is not None and len(st.session_state.search_messages) == 1:
        df = pd.read_csv(uploaded_file)
        embedding = Embedding(df)
        embedding_data = embedding.do_embedding()
        search = Search(embedding_data)
    elif uploaded_file is not None and len(st.session_state.search_messages) != 1:
        df = pd.read_csv(uploaded_file)
        embedding = Embedding(df)
        embedding_data = embedding.load_data()
        search = Search(embedding_data)
        
    st.subheader("웹툰을 검색해보세요!")
    st.caption("ex) 주인공이 못생겼다가 예뻐진 웹툰이 뭐지? \n 주인공이 힘순찐인 웹툰 알려줘 \n 수호라는 캐릭터가 등장하는 웹툰 알려줘 \n 무림 웹툰 추천해줘")
    
    if "search_messages" not in st.session_state:
        st.session_state["search_messages"] = [{"role": "assistant", "content": "안녕하세요. 웹툰에 대한 질문을 해주세요!"}]

    for msg in st.session_state.search_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input_key = "search_chat_input_sanghoon"
    # 사용자 인풋 받기  
    if prompt := st.chat_input("웹툰을 검색해보세요", key=chat_input_key):
        # 사용자 입력 보여 주기
        st.session_state.search_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 대화 보여 주기
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = search.receive_chat(prompt)
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.search_messages.append({"role": "assistant", "content": assistant_response})
        
        
def character_page():
    
    st.title("이 웹툰 뭐였지 Demo Page")
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    
    if uploaded_file is not None:
        character = Character(uploaded_file)    
    
    st.subheader("캐릭터와 대화 해보세요!")
    st.caption("ex) 교회를 어떻게 생각해?\n너에게 엄마란 무슨 존재야?")
    
    if "character_messages" not in st.session_state:
        st.session_state["character_messages"] = []

    for msg in st.session_state.character_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    chat_input_key = "character_chat_input"
    # 사용자 인풋 받기
    if prompt := st.chat_input("캐릭터에게 할 말을 입력하세요.", key=chat_input_key):
        # 사용자 입력 보여 주기
        st.session_state.character_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # 봇 대화 보여 주기
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = character.receive_chat(prompt)
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.character_messages.append({"role": "assistant", "content": assistant_response})
    
if __name__ == "__main__":
    main()