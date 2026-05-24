from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
from typing import Dict, Any, Tuple, List
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

##loading text file from the directory
dir_loader = DirectoryLoader(
    "data/pdf",
    glob = "**/*.pdf",
    loader_cls = PyMuPDFLoader,
)
pdf_documents = dir_loader.load()
pdf_documents

# CHUNKING (DIVIDING LARGE FILES INTO SMALLER PIECES)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
    separators = ["\n\n", "\n", ".", " ", ""]
)

chunks = text_splitter.split_documents(pdf_documents)

# embedding and vectorStoreDB

class EmbeddingManager:
    """Handels Document embedding generation using sentenceTransformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initializing the embedding manager
        Args:
            model_name: HuggingFace Model for sentence embeddings    
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loading the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully: Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model{self.model_name}: {e}")
            raise


    def generate_embeddings(self, texts: list[str])->np.ndarray:
        """
        Generate embeddings for a list of texts

        args:
            Texts: list of string to embed

        Returns: 
                numpy array of embeddings with shape (len(texts), embedding_dim)
        """


        if not self.model:
            raise ValueError("Model not loaded")
        else:
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar = True)
            print(f"Generated Embeddings with shape: {embeddings.shape}")
            return embeddings
        


#Initializing the embedding manager
embedding_manager = EmbeddingManager()

# VectorStore
class VectorStore:
    """Manages Document embeddings in a chromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str  = "data/vectorStor"):
        """Initialize the vector store
        args:
            collection_name: Name of the chromaDB collection
            persist_directory: Directory to persist the vector store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the chromaDB client and collection"""
        try:
            #create persistent chromadb client
            os.makedirs(self.persist_directory, exist_ok = True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            #get or create collection
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store intialized collection:  {self.collection_name}")
            print(f"Existing document in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error Initializing vector store: {e}")
            raise
            
    def add_documents(self, documents: list[Any], embeddings: np.ndarray):
        """Add documents and their embedding to the vector store
        Args:
            documents: List of Langchain documents
            embeddings: corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings): 
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")

        #prepare data for chromadb
        ids =  []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            #Generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            #document content
            documents_text.append(doc.page_content)

            #embedding
            embeddings_list.append(embedding.tolist())

        #add to collection
        try:
            self.collection.add(
                ids = ids,
                metadatas = metadatas,
                embeddings = embeddings_list,
                documents = documents_text
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print("Error adding document to vector store: {e}")
            raise

vector_store =VectorStore()

if vector_store.collection.count() == 0:
    #converting text into embeddings
    texts = [doc.page_content for doc in chunks]

    #generating the embeddings
    embeddings  = embedding_manager.generate_embeddings(texts)

    #storing into the vector database
    vector_store.add_documents(chunks, embeddings)

# Retriever Pipeline from vector
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retireiver
        Args:
            Vector_store: vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold:  float =0.0)-> List[Dict[str,Any]]:
        """
        Retrieve relevant document for a  query
        Args:
            query: The search  query
            top_k: Number of top results to return
            score_threshold:Minimum similarity score threshold
        Returns:
            List of dictionaries containing retrieved documents and metadata    
        """

        print(f"Retreiving documents for query: {query}")
        print(f"Top K: {top_k}, score_threshold: {score_threshold}")

        #generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        #search in a vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k
            )

            #process results
            retrieved_docs = []

            seen_content = set()  # To track seen document content and avoid duplicates


            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #converting distance to similarity score(chromadb uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score < score_threshold:
                        continue

                    # deduplicating overlapping chunks
                    content_preview = document[:200].strip()
                    if content_preview in seen_content:
                        continue

                    seen_content.add(content_preview)
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': len(retrieved_docs) + 1
                    })
                    if len(retrieved_docs) >= top_k:
                        break

                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No documents found")

            return retrieved_docs
        
        except Exception as e:
            print("Error during retrieval: {e}")
            return[]

rag_retreiver = RAGRetriever(vector_store, embedding_manager)

# Integration Vectordb context pipeline with LLM output
#simple RAG pipeline with GROQ LLM
load_dotenv()

#initiaize the groq llm
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.3-70b-versatile", temperature=0.1, max_tokens = 1024)

#simple rag function: retrieve context + generate response
def rag_simple(query, retriever, llm, top_k = 3):
    #retrieve the context
    results = retriever.retrieve(query, top_k = top_k)
    context = "\n\n".join([doc['content'] for doc in  results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    

    #generate the answer using groq llm
    prompt = f"""Use the following context to answer the question concisely.
            context:{context}
            Question: {query}
            Answer:"""
    
    response = llm.invoke([prompt.format(context = context, query = query)])
    return response.content


def rag_advanced(query, retriever, llm, top_k = 5, min_score = 0.2, return_context = False):
    """
    RAG Pipeline with extra features: 
    -Returns answer, sources, confidence score and optionally full context.
    """

    results = retriever.retrieve(query, top_k = top_k, score_threshold = min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [],'confidence': 0.0, 'context': ''}
    #prepare context and sources
    context = "\n\n".join([doc['content'] for doc in  results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    }for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])

    #generate answer
    prompt = f"""Use the following context to answer the question precisely. \ncontext: \n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context = context, query= query)])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context
    return output







