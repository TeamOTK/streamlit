from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from langchain_community.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain_community.vectorstores import Chroma,  Qdrant, FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from qdrant_client import QdrantClient
import time
from typing import List
from pydantic import BaseModel, Field
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_KSW")
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
        """ Get data from csv file """
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
        """ Get data from web """
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    
    def get_multi_data(self, url_list):
        """ Get multi data from web """
        docs = []
        for url in url_list:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            print(url)
        return docs
        
    def get_text_splitter(self, data):
        """ Split text """
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        documents = text_splitter.split_documents(data)
        return documents
    
    def get_cached_embedder(self):
        """ Get cached embedder -> Speed up """""
        underlying_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        store = LocalFileStore("./cache/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        return cached_embedder
    
    def get_embeddings(self, documents ,cached_embedder):
        # client = QdrantClient()
        # collection_name = "MyCollection"
        # qdrant = Qdrant(client, collection_name, embeddings)
        # db = await qdrant.from_documents(documents, embeddings, "http://localhost:6333", collection_name)
        db = FAISS.from_documents(documents, cached_embedder)
        return db
    
    def get_embeddings_filter(self):
        embeddings = OpenAIEmbeddings()
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        return embeddings_filter
    
    def get_output_parser(self):
        output_parser = LineListOutputParser()
        return output_parser
    
    def get_retriever(self, db):
        """ Create a retriever from the vectorstore """
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.8,
                "k": 3
        })
        return retriever
    
    def get_muti_query_retriever(self, db, llm_chain):
        """ Muti query retriever for use LLM Chain """
        retriever = MultiQueryRetriever(
            retriever=db.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output
        return retriever
    
    def get_pipeline_compression_retriever(self, retriever, embeddings):
        
        """ Create a pipeline of document transformers and a retriever """
        
        ## filters
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def get_parent_document_retriever(self, embeddings):
        """ Create a parent document retriever for chunked documents """
        # This text splitter is used to create the child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=embeddings
        )
        # The storage layer for the parent documents
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        return retriever, vectorstore
    
    def add_docs_to_retriever(self, retriever, docs):
        retriever.add_documents(docs, ids=None)
        return retriever
            
    def get_all_relevant_documents(self, query, retriever):
        # Get relevant documents ordered by relevance score
        docs = retriever.get_relevant_documents(query)
        
        result = []
        for doc in docs:
            # result.append(doc.page_content.split("\n")[1])
            result.append(doc.metadata["title"])
        return result
    
    def get_sub_relevant_documents(self, query, vectorstore):
        sub_docs = vectorstore.similarity_search(query)
        return sub_docs[0].metadata["title"]

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
        
    def make_retriever(self, file_path):
        data = self.get_data(file_path)
        documents = self.get_text_splitter(data)
        # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        cached_embedder = self.get_cached_embedder()
        db = self.get_embeddings(documents, cached_embedder)
        # retriever = self.get_parent_document_retriever(embeddings)
        # for url in namu_list:
        #     docs = self.get_web_data(url)
        #     retriever = self.add_docs_to_retriever(retriever, docs)
        retriever = self.get_retriever(db)
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(retriever, cached_embedder)
        return pipeline_compression_retriever
    
    def make_retriever_from_url(self, url_list):
        cached_embedder = self.get_cached_embedder()
        # db = self.get_embeddings(docs, cached_embedder)
        # retriever = self.get_retriever(db)
        retriever, vectorstore = self.get_parent_document_retriever(cached_embedder)
        docs = self.get_multi_data(url_list)
        print(docs)
        retriever.add_documents(docs, ids=None)
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(retriever, cached_embedder)
        return pipeline_compression_retriever, vectorstore
    
    def run(self, query, pipeline_compression_retriever, vectorstore):
        result = self.get_all_relevant_documents(query, pipeline_compression_retriever)
        # result = self.get_sub_relevant_documents(query, pipeline_compression_retriever, vectorstore)
        print(result)

if __name__ == "__main__":
    search = Search()
    query = "고딩들이 농구하는 웹툰 알려줘"
    
    file_name = "namu_new.csv"
    file_path = os.path.join("..", "streamlit/src/model/data/webtoon", file_name)

    absolute_file_path = os.path.abspath(file_path)
    # search.run(query, absolute_file_path)
    ## 화산귀환, 신의탑, 외모지상주의, 나이트런, 전지적 독자 시점, 재혼황후, 가비지타임, 내일, 삼국지톡, 놓지마 정신줄
    namu_list = [
        # "https://namu.wiki/w/%ED%99%94%EC%82%B0%EA%B7%80%ED%99%98(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EC%8B%A0%EC%9D%98%20%ED%83%91",
        # "https://namu.wiki/w/%EC%99%B8%EB%AA%A8%EC%A7%80%EC%83%81%EC%A3%BC%EC%9D%98(%EC%9B%B9%ED%88%B0)",
        # "https://namu.wiki/w/%EB%82%98%EC%9D%B4%ED%8A%B8%EB%9F%B0",
        "https://namu.wiki/w/%EC%A0%84%EC%A7%80%EC%A0%81%20%EB%8F%85%EC%9E%90%20%EC%8B%9C%EC%A0%90(%EC%9B%B9%ED%88%B0)",
        # "https://namu.wiki/w/%EC%9E%AC%ED%98%BC%20%ED%99%A9%ED%9B%84(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EA%B0%80%EB%B9%84%EC%A7%80%ED%83%80%EC%9E%84",
        # "https://namu.wiki/w/%EB%82%B4%EC%9D%BC(%EC%9B%B9%ED%88%B0)",
        # "https://namu.wiki/w/%EC%82%BC%EA%B5%AD%EC%A7%80%ED%86%A1",
        # "https://namu.wiki/w/%EB%86%93%EC%A7%80%EB%A7%88%20%EC%A0%95%EC%8B%A0%EC%A4%84"
    ]
    now = time.time()
    pipeline_compression_retriever, vectorstore = search.make_retriever_from_url(namu_list)
    # pipeline_compression_retriever = search.make_retriever(file_path)
    print(time.time()- now)
    now2 = time.time()
    search.run(query, pipeline_compression_retriever, vectorstore)
    print(time.time()- now2)
