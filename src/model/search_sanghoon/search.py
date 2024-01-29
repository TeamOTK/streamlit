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
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

import time
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
            'fieldnames': ['title', 'author', 'genre', 'description', 'character']},
            encoding = 'UTF-8'
        )
        data = loader.load()
        return data
    
    def get_web_data(self, url):
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    
    def get_multi_data(self, url_list):
        docs = []
        for url in url_list:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
            print(url)
        return docs
        
    def get_text_splitter(self, data):
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        documents = text_splitter.split_documents(data)
        return documents
    
    def get_cached_embedder(self):
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
        # Create a retriever
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,
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
    
    def get_pipeline_compression_retriever(self, retriever, embeddings):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        
        # retriever = db.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={
        #         "k": 3
        #     },
        # )
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )
        return compression_retriever
    
    def get_parent_document_retriever(self, embeddings):
        # This text splitter is used to create the child documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
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
            parent_splitter=parent_splitter,
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
            # result.append(doc.page_content.split("\n")[1])
            result.append(doc.metadata["title"])
        return result
        
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
        docs = self.get_multi_data(url_list)
        # db = self.get_embeddings(docs, cached_embedder)
        # retriever = self.get_retriever(db)
        retriever = self.get_parent_document_retriever(cached_embedder)
        # print(docs)
        retriever.add_documents(docs, ids=None)
        pipeline_compression_retriever = self.get_pipeline_compression_retriever(retriever, cached_embedder)
        return pipeline_compression_retriever
    
    def run(self, query, pipeline_compression_retriever):
        result = self.get_all_relevant_documents(query, pipeline_compression_retriever)
        print(result)
        

if __name__ == "__main__":
    search = Search()
    query = "저승을 주제로 한 웹툰 알려줘"
    
    file_name = "namu.csv"
    file_path = "./namu.csv"

    absolute_file_path = os.path.abspath(file_path)
    # search.run(query, absolute_file_path)
    ## 화산귀환, 신의탑, 외모지상주의, 나이트런, 전지적 독자 시점, 재혼황후, 가비지타임, 내일, 삼국지톡, 놓지마 정신줄
    namu_list = [
        "https://namu.wiki/w/%ED%99%94%EC%82%B0%EA%B7%80%ED%99%98(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EC%8B%A0%EC%9D%98%20%ED%83%91",
        "https://namu.wiki/w/%EC%99%B8%EB%AA%A8%EC%A7%80%EC%83%81%EC%A3%BC%EC%9D%98(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EB%82%98%EC%9D%B4%ED%8A%B8%EB%9F%B0",
        "https://namu.wiki/w/%EC%A0%84%EC%A7%80%EC%A0%81%20%EB%8F%85%EC%9E%90%20%EC%8B%9C%EC%A0%90(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EC%9E%AC%ED%98%BC%20%ED%99%A9%ED%9B%84(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EA%B0%80%EB%B9%84%EC%A7%80%ED%83%80%EC%9E%84",
        "https://namu.wiki/w/%EB%82%B4%EC%9D%BC(%EC%9B%B9%ED%88%B0)",
        "https://namu.wiki/w/%EC%82%BC%EA%B5%AD%EC%A7%80%ED%86%A1",
        "https://namu.wiki/w/%EB%86%93%EC%A7%80%EB%A7%88%20%EC%A0%95%EC%8B%A0%EC%A4%84"
    ]
    now = time.time()
    pipeline_compression_retriever = search.make_retriever_from_url(namu_list)
    # pipeline_compression_retriever = search.make_retriever(file_path)
    print(time.time()- now)
    now2 = time.time()
    search.run(query, pipeline_compression_retriever)
    print(time.time()- now2)
