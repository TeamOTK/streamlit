import streamlit as st
import time
import os
import pandas as pd

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

from model.character.character import Character
from model.search.search import Search

def main():
    selected_page = st.sidebar.selectbox("Select a page", ["웹툰 검색하기", "캐릭터와 대화하기"])

    if selected_page == "웹툰 검색하기":
        search_page()
    elif selected_page == "캐릭터와 대화하기":
        character_page()

def search_page():
    st.title("이 웹툰 뭐였지 Demo Page")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        search = Search(df)
        
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
        
def character_page():
    
    st.title("이 웹툰 뭐였지 Demo Page")
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    
    st.subheader("캐릭터와 대화 해보세요!")
    st.caption("ex) 교회를 어떻게 생각해?\n너에게 엄마란 무슨 존재야?")
    
    character = Character(uploaded_file)    
    
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