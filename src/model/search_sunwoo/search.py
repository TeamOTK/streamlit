from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

class Search():
    
    def __init__(self):
        pass
    
    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    
    def get_retriever(self, texts, embeddings):
        # Create a retriever
        retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
            search_kwargs={"k": 10}
        )
        return retriever
        
    # texts = [
    #     "Basquetball is a great sport.",
    #     "Fly me to the moon is one of my favourite songs.",
    #     "The Celtics are my favourite team.",
    #     "This is a document about the Boston Celtics",
    #     "I simply love going to the movies",
    #     "The Boston Celtics won the game by 20 points",
    #     "This is just a random text.",
    #     "Elden Ring is one of the best games in the last 15 years.",
    #     "L. Kornet is one of the best Celtics players.",
    #     "Larry Bird was an iconic NBA player.",
    # ]
    
    def get_relevant_documents(self, query, retriever):
        # Get relevant documents ordered by relevance score
        docs = retriever.get_relevant_documents(query)
        
        # Reorder the documents:
        # Less relevant document will be at the middle of the list and more
        # relevant elements at beginning / end.
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)
        return reordered_docs

    
    # query = "What can you tell me about the Celtics?"

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
        
    def run(self, query, texts):
        embeddings = self.get_embeddings()
        retriever = self.get_retriever(texts, embeddings)
        reordered_docs = self.get_relevant_documents(query, retriever)
        chain = self.get_stuff_chain()
        chain.run(input_documents=reordered_docs, query=query)
        
if __name__ == "__main__":
    search = Search()
    texts = [
        "Basquetball is a great sport.",
        "Fly me to the moon is one of my favourite songs.",
        "The Celtics are my favourite team.",
        "This is a document about the Boston Celtics",
        "I simply love going to the movies",
        "The Boston Celtics won the game by 20 points",
        "This is just a random text.",
        "Elden Ring is one of the best games in the last 15 years.",
        "L. Kornet is one of the best Celtics players.",
        "Larry Bird was an iconic NBA player.",
    ]
    query = "What can you tell me about the Celtics?"
    search.run(query, texts)