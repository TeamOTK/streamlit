from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.vectorstores import Chroma, Qdrant, FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import OpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from qdrant_client import QdrantClient

from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_HSH")
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self):
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

class Search():
    
    def __init__(self):
        pass     
    
    def get_data(self, file_path):
        loader = CSVLoader(
            file_path=file_path,
            csv_args={
            # 'delimiter': ',',
            # 'quotechar': '"',
            'fieldnames': ['id', 'title', 'genre', 'description']},
            encoding = 'UTF-8'
        )
        data = loader.load()
        return data
    
    def get_web_data(self, url):
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    
    def get_multi_data(self, loaders):
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
        docs = text_splitter.split_documents(docs)
        return docs
        
    def get_text_splitter(self, data):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(data)
        return documents
    
    def get_embeddings (self, documents ,embeddings):
        # client = QdrantClient()
        # collection_name = "MyCollection"
        # qdrant = Qdrant(client, collection_name, embeddings)
        # db = await qdrant.from_documents(documents, embeddings, "http://localhost:6333", collection_name)
        db = FAISS.from_documents(documents, embeddings)
        return db
    
    def get_embeddings_filter(self):
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        return embeddings_filter
    
    def get_output_parser(self):
        output_parser = LineListOutputParser()
        return output_parser
    
    def get_retriever(self, db):
        # Create a retriever
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.5,
                "k": 3
        })
        return retriever
    
    def get_muti_query_retriever(self, db, llm_chain):
        # Run
        retriever = MultiQueryRetriever(
            retriever=db.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output
        return retriever
        
    def get_compression_retriever(self, db, embeddings_filter):
        # Create a retriever
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.5,
                "k": 3
            },
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, base_retriever=retriever
        )
        return compression_retriever
    
    def get_pipeline_compression_retriever(self, db, embeddings):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3
            },
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def get_parent_document_retriever(self):
        # This text splitter is used to create the child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=OpenAIEmbeddings()
        )
        # The storage layer for the parent documents
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        return retriever
    
    def add_docs_to_retriever(self, retriever, docs):
        retriever.add_documents(docs, ids=None)
        return retriever
            
    def get_all_relevant_documents(self, query, retriever):
        # Get relevant documents ordered by relevance score
        docs = retriever.get_relevant_documents(query)
        
        # Reorder the documents:
        # Less relevant document will be at the middle of the list and more
        # # relevant elements at beginning / end.
        # reordering = LongContextReorder()
        # reordered_docs = reordering.transform_documents(docs)
        result = []
        for doc in docs:
            result.append(doc.page_content.split("\n")[1])
        return result
    
    def get_sub_relevant_documents(self, query, vectorstore):
        sub_docs = vectorstore.similarity_search(query)
        return sub_docs[0].page_content

    # query = "What can you tell me about the Celtics?"
    
    def get_llm_chain(self, output_parser):
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            You must answer in Korean.
            Original question: {question}""",
        )
        llm = ChatOpenAI(temperature=0)

        # Chain
        llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
        return llm_chain

    def get_stuff_chain(self):
        # Override prompts
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        document_variable_name = "context"
        llm = OpenAI()
        stuff_prompt_override = """Given this text extracts:
        -----
        {context}
        -----
        Please answer the following question:
        {query}"""
        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query"]
        )
        # Instantiate the chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        return chain
        
    def run(self, query, file_path):
        data = self.get_data(file_path)
        documents = self.get_text_splitter(data)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = self.get_embeddings(documents, embeddings)
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(db, embeddings)
        result = self.get_all_relevant_documents(query, pipeline_compression_retriever)
        print(result)
        

if __name__ == "__main__":
    search = Search()
    query = "매화검존 청명이 회귀하여 화산파를 성장시키는 내용의 웹툰"
    
    file_name = "namu_new.csv"
    file_path = os.path.join("..", "streamlit/src/model/data/webtoon", file_name)

    absolute_file_path = os.path.abspath(file_path)
    search.run(query, absolute_file_path)
