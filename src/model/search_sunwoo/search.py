from langchain.storage import InMemoryStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages

from langchain_community.document_transformers import  EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma,  Qdrant, FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain.text_splitter import CharacterTextSplitter 

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

import time
from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_KSW")
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )

class Search():
    
    def __init__(self):
        pass     
    
    def get_data_from_csv(self, file_path):
        """ Get data from csv file """
        loader = CSVLoader(
            file_path=file_path,
            # csv_args={
            # 'delimiter': ',',
            # 'quotechar': '"',
            # 'fieldnames': ['id', 'title', 'genre', 'description']},
            encoding = 'UTF-8'
        )
        data = loader.load()
        return data
    
    def get_data_from_web(self, url):
        """ Get data from web """
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    
    def get_multi_data_from_dataframe(self, datas):
        """ Get multi data  """
        docs = []
        for data in datas:
            loader = DataFrameLoader(data, page_content_column="description")
            for i in loader.lazy_load():
                docs.append(i)
        return docs
        
    def get_text_splitter(self, docs):
        """ Split text """
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)
        return documents
    
    def get_cached_embedder(self):
        """ Get cached embedder -> Speed up """""
        underlying_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        store = LocalFileStore("./cache/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        return cached_embedder
    
    def get_embeddings(self, documents, cached_embedder, collection_name="webtoon"):
        vectorstore = Chroma.from_documents(documents, cached_embedder,
            collection_name=collection_name)
        return vectorstore
    
    def get_retriever(self, db):
        """ Create a retriever """
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,
                "k": 1
        }
        )
        return retriever
    
    def get_pipeline_compression_retriever(self, retriever, embeddings):
        """ Create a pipeline of document transformers and a retriever """
        ## filters
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.8)
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def get_retriever_tool(self, retriever):
        """ Create a retriever tool """
        retriever_tool = create_retriever_tool(
            retriever,
            "webtoon-retriever",
            "Query a retriever to get information about webtoon",
        )
        return retriever_tool
    
    def parse(self, output):
        # If no function was invoked, return to user
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"output": output.content}, log=output.content)

        # Parse out the function call
        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])

        # If the Response function was invoked, return to the user with the function inputs
        if name == "Response":
            return AgentFinish(return_values=inputs, log=str(function_call))
        # Otherwise, return an agent action
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
        )
            
    def get_agent(self, retriever_tool):
        system_message = """
        You are an AI responding to users searching for webtoons. 
        Summarize the data in three lines or less and provide a friendly response in Korean.
        data: {data}
        
        You always follow these guidelines:
            -If the answer isn't available within the context, state that fact
            -Otherwise, answer to your best capability, refering to source of documents provided
            -Only use examples if explicitly requested
            -Do not introduce examples outside of the context
            -Do not answer if context is absent
            -Limit responses to three or four sentences for clarity and conciseness
            -You must answer in Koreans
        """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0, 
            openai_api_key=OPENAI_API_KEY
        )
        
        llm_with_tools = llm.bind_functions([retriever_tool, Response])
        
        agent = (
            {
                "data": lambda x: x["data"],
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | self.parse
        )
        
        agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)
        return agent_executor
    
    def add_docs_to_retriever(self, retriever, docs):
        retriever.add_documents(docs, ids=None)
        return retriever
            
    def get_all_relevant_documents(self, query, retriever):
        # Get relevant documents ordered by relevance score
        docs = retriever.get_relevant_documents(query)
        
        # result = []
        # for doc in docs:
        #     # result.append(doc.page_content)
        #     result.append(doc)
        # return result
        return docs[0].page_content
    
    def get_sub_relevant_documents(self, query, vectorstore):
        sub_docs = vectorstore.similarity_search(query)
        return sub_docs[0].metadata["title"]
    
    def make_retriever(self, datas):
        docs = self.get_multi_data_from_dataframe(datas)
        # print(docs)
        documents = self.get_text_splitter(docs)
        
        cached_embedder = self.get_cached_embedder()
        db = self.get_embeddings(documents, cached_embedder)
        
        retriever = self.get_retriever(db)
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(retriever, cached_embedder)
        return pipeline_compression_retriever
    
    def run(self, query, pipeline_compression_retriever):
        retriever_tool = self.get_retriever_tool(pipeline_compression_retriever)
        result = self.get_all_relevant_documents(query, pipeline_compression_retriever)
        
        agent_executor = self.get_agent(retriever_tool) 
        result = agent_executor(
            {   "data": result,
                "input": query},
            return_only_outputs=True)
        return result

if __name__ == "__main__":
    search = Search()
    query = "재혼하는 웹툰 찾아줘"
    
    file_name = "namu_new.csv"
    file_path = os.path.join("..", "streamlit/src/model/data/webtoon", file_name)

    absolute_file_path = os.path.abspath(file_path)
    df = pd.read_csv(absolute_file_path)
    # print(df)
    datas = []
    datas.append(df)
    ## 화산귀환, 신의탑, 외모지상주의, 나이트런, 전지적 독자 시점, 재혼황후, 가비지타임, 내일, 삼국지톡, 놓지마 정신줄
    now = time.time()
    # pipeline_compression_retriever, vectorstore = search.make_retriever_from_url(namu_list)
    pipeline_compression_retriever = search.make_retriever(datas)
    print("Make Retriever: " + str(time.time()- now))
    now2 = time.time()
    result = search.run(query, pipeline_compression_retriever)
    print("Find result: " + str(time.time()- now2))
    print(result)
