from sentence_transformers import SentenceTransformer
import os

from chromadb_client import ChromaDBClient
from chunk_code import extract_code_chunks
from qwencoderllm import analyse_file

class Embedding:

    def __init__(self):
        self.metadata_store = {}
        self.db_client = ChromaDBClient(embed_model_name="sentence-transformers/all-mpnet-base-v2")
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
                        
                    for chunk in chunks:
                    #    chunk_summary = await analyse_file(chunk)
                        self.db_client.add_text(
                            name,
                            str(chunk_id),
                            "",
                            {"file":file})
                        chunk_id += 1
        
        #embeddings = self.embedding_model.encode(code_chunks, convert_to_numpy = True)
        #dimension = embeddings.shape[1]
        #index = faiss.IndexFlatL2(dimension)
        #index.add(embeddings)

        #os.makedirs(self.vector_store_dir, exist_ok = True)
        #faiss.write_index(index, os.path.join(self.vector_store_dir, "faiss_index"))

        #self.vector_store = index


    def retrieve_similar_code(self, collection_name, query, top_k=5):
        self.db_client.query_text(collection_name, query, top_k)

    def retrieve_code_for_microservice(self, name):
        self.db_client.query_text(name, name)