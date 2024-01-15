import os
import pandas as pd
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.types import AgentType

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

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

llm = ChatOpenAI(
    model_name="gpt-4-1106-preview",  # Name of the language model
    temperature=0  # Parameter that controls the randomness of the generated responses
)

embeddings = OpenAIEmbeddings()


def load_csv(filepath):
    return pd.read_csv(filepath)


def format_data_for_gpt(data):
    formatted_data = []
    for index, row in data.iterrows():
        formatted_data.append(
            f"title: {row['title']} author: {row['author']} genre: {row['genre']} description: {row['description']}\n")
    return formatted_data


loader = load_csv("../data/webtoon/info.csv")
docs = format_data_for_gpt(loader)

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

query = "주인공이 힘순찐인 웹툰 알려줘"
result = agent(query)
print(result['output'])
