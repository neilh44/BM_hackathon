from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
import logging
import shutil

class VectorStore:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            filename=f'vectorstore_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
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
        self.active_collection = None
        
        # Create persist directory if it doesn't exist
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
            
        self.logger.info("VectorStore initialized successfully")

    def cleanup_previous_collection(self):
        """Clean up the previous collection if it exists"""
        try:
            if self.active_collection:
                self.logger.info(f"Cleaning up previous collection: {self.active_collection}")
                db = Chroma(
                    persist_directory=self.persist_dir,
                    collection_name=self.active_collection,
                    embedding_function=self.embeddings
                )
                db.delete_collection()
                # Clean up collection-specific files
                collection_dir = os.path.join(self.persist_dir, self.active_collection)
                if os.path.exists(collection_dir):
                    shutil.rmtree(collection_dir)
                self.logger.info(f"Successfully cleaned up collection: {self.active_collection}")
        except Exception as e:
            self.logger.error(f"Error cleaning up collection: {str(e)}")
            # Continue with new collection creation even if cleanup fails
    
    def load_to_vectorstore(self, markdown_path):
        try:
            self.logger.info(f"Loading document from {markdown_path}")
            
            # Clean up previous collection
            self.cleanup_previous_collection()
            
            # Create new collection
            loader = TextLoader(markdown_path)
            docs = self.text_splitter.split_documents(loader.load())
            
            collection_name = f"pdf_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Creating collection: {collection_name}")
            
            vectorstore = Chroma.from_documents(
                docs,
                self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=collection_name
            )
            
            # Update active collection
            self.active_collection = collection_name
            
            self.logger.info(f"Successfully loaded document into collection {collection_name}")
            return collection_name
        except Exception as e:
            self.logger.error(f"Vector storage error: {str(e)}")
            raise

    def get_context(self, query: str, collection_name: str) -> str:
        """Get relevant context without LLM processing"""
        try:
            if collection_name != self.active_collection:
                raise ValueError("Invalid or expired collection")
                
            vectorstore = Chroma(
                persist_directory=self.persist_dir,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            context_docs = vectorstore.similarity_search(query, k=5)
            return "\n".join(doc.page_content for doc in context_docs)
        except Exception as e:
            self.logger.error(f"Error getting context: {str(e)}")
            raise

    def query_document(self, query: str, collection_name: str) -> str:
        """Query document with LLM processing"""
        try:
            if collection_name != self.active_collection:
                raise ValueError("Invalid or expired collection. Please re-upload the document.")
                
            self.logger.info(f"Querying collection {collection_name} with: {query}")
            context = self.get_context(query, collection_name)
            
            self.logger.info("Sending query to Groq LLM")
            prompt = f"Answer based on this context:\n{context}\n\nQuestion: {query}"
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            )
            
            self.logger.info("Successfully received response from LLM")
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            raise

    def query_document_raw(self, prompt: str, collection_name: str) -> str:
        """Send raw prompt to LLM without additional processing"""
        try:
            if collection_name != self.active_collection:
                raise ValueError("Invalid or expired collection")
                
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.2  # Lower temperature for more precise extraction
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Raw query error: {str(e)}")
            raise

    def cleanup(self):
        """Clean up all collections and persist directory"""
        try:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
            self.logger.info("Cleaned up all collections and persist directory")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise