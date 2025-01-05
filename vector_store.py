from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
import logging

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.persist_dir = "chroma_db"

    def load_to_vectorstore(self, markdown_path):
        try:
            loader = TextLoader(markdown_path)
            docs = self.text_splitter.split_documents(loader.load())
            
            collection_name = f"pdf_docs_{datetime.now().strftime('%Y%m%d')}"
            
            vectorstore = Chroma.from_documents(
                docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=collection_name
            )
            
            return collection_name
        except Exception as e:
            self.logger.error(f"Vector storage error: {str(e)}")
            raise

    def query_document(self, query, collection_name):
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            context_docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join(doc.page_content for doc in context_docs)
            
            prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}"
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            raise