import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import numpy as np
#import faiss-cpu
from chromadb_client import ChromaDBClient
from chunk_code import extract_code_chunks
from qwencoderllm import analyse_file

class Embedding:

    def __init__(self):
        self.metadata_store = {}
        #self.db_client = ChromaDBClient(embed_model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Initialize the embedding model
        embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize FAISS index for vector database
        #index = faiss-cpu.IndexFlatL2(self.embedding_model.get_sentence_embedding_dimension())
        self.graph = nx.Graph()
        #self.db_client = ChromaDBClient(embed_model_name="sentence-transformers/all-mpnet-base-v2")
        #self.vector_store_dir = os.path.join(os.getcwd(), "vector_store")
        #self.vector_store = None
        #self.db_client.create_collection("code_collection",{"desc":"test"})
        #if(os.path.exists(self.vector_store_dir)):
        #    self.vector_store = faiss.read_index(os.path.join(self.vector_store_dir, "faiss_index"))

    async def add_dotnet_codebase_embeddings(self, name, directory, language):
        """Walk through a .NET project, chunk and embed code."""
        chunk_id = 0
        global code
        
        code_chunks = []

        chunk_id = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith("." + language):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    
                    chunks = [code]
                    if language == 'cs':
                        chunks = extract_code_chunks(code)
                        
                    #for chunk in chunks:
                    #    chunk_summary = await analyse_file(chunk)
                        """    self.db_client.add_text(
                            name,
                            str(chunk_id),
                            "",
                            {"file":file})
                        chunk_id += 1
                    """
                    embedding = self.embedding_model.encode(code)

                # Store node with embedding
                    self.graph.add_node(os.path.join(root, file), embedding=embedding, content=code)
        #embeddings = self.embedding_model.encode(code_chunks, convert_to_numpy = True)
        #dimension = embeddings.shape[1]
        #index = faiss.IndexFlatL2(dimension)
        #index.add(embeddings)

        #os.makedirs(self.vector_store_dir, exist_ok = True)
        #faiss.write_index(index, os.path.join(self.vector_store_dir, "faiss_index"))

        #self.vector_store = index

    def GetFileContent(file_path):
        """
        Read the content of a file.
        """
        with open(file_path, "r") as file:
            return file.read()
        
    def GetRelevantContext(self, file_content, top_k=5):
        query_embedding = self.embedding_model.encode(file_content).reshape(1, -1)
        
        similarities = {
            node: cosine_similarity(query_embedding, data["embedding"].reshape(1, -1))[0][0]
            for node, data in self.graph.nodes(data=True)
        }
        
        top_nodes = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
        context_contents = [self.graph.nodes[node]["content"] for node in top_nodes]
        
        return "\n".join(context_contents)
