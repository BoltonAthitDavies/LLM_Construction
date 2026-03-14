# build rag framework
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

class RAG:
    def __init__(self, documents_path: str, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.documents_path = documents_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self._create_vector_store()
        self.llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def _create_vector_store(self):
        # Load documents
        loader = TextLoader(self.documents_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Create vector store
        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store

    def answer_query(self, query: str) -> str:
        return self.qa_chain.run(query)

# Example usage:
# rag = RAG(documents_path="path/to/your/documents.txt", openai_api_key="your_openai_api_key")
# response = rag.answer_query("Your question here")
