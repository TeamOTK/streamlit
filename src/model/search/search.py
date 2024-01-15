from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.types import AgentType

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_HSH")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

class Search:
    def __init__(self, uploaded_file):
        ## 배포용
        self.uploaded_file = uploaded_file
        
    def get_agent(self): # GPT-4를 사용하여 대화를 생성하는 코드
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",  # Name of the language model
            openai_api_key=OPENAI_API_KEY,
            temperature=0  # Parameter that controls the randomness of the generated responses
        )
        
        system_message = """
        You always follow these guidelines:

                -If the answer isn't available within the context, state that fact
                -Otherwise, answer to your best capability, refering to source of documents provided
                -Only use examples if explicitly requested
                -Do not introduce examples outside of the context
                -Do not answer if context is absent
                -Limit responses to three or four sentences for clarity and conciseness
                -You must answer in Koreans
        """
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        file_name = "info.csv"
        file_path = os.path.join("..", "streamlit/src/model/data/webtoon", file_name)

        absolute_file_path = os.path.abspath(file_path)
        
        loader = self.load_csv(absolute_file_path)
        ## 로컬
        # docs = self.format_data_for_gpt(loader)
        
        ## 배포용
        docs = self.format_data_for_gpt(self.uploaded_file)

        vectorstore = FAISS.from_texts(docs, embeddings)

        retriever = vectorstore.as_retriever()

        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key='input', return_messages=True, output_key='output')

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            verbose=True,
            return_source_documents=True
        )

        tools = [
            Tool(
                name="doc_search_tool",
                func=qa,
                description=(
                    "This tool is used to retrieve information from the knowledge base"
                )
            )
        ]

        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            memory=memory,
            return_source_documents=True,
            return_intermediate_steps=True,
            agent_kwargs={"system_message": system_message}
        )
        
        return agent


    def load_csv(self, filepath):
        return pd.read_csv(filepath)

    def format_data_for_gpt(self, data):
        formatted_data = []
        for index, row in data.iterrows():
            formatted_data.append(
                f"title: {row['title']} author: {row['author']} genre: {row['genre']} description: {row['description']}\n")
        return formatted_data
    
    def receive_chat(self, query):
        agent = self.get_agent()
        result = agent(query)
        return result['output']

