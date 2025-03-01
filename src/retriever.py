from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load Embedding Model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load stored ChromaDB
db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

# Function to retrieve relevant chunks
def get_relevant_docs(query, k=1):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Test the retriever
query = "what are the types of financial markets?"
retrieved_docs = get_relevant_docs(query)

print("📌 Retrieved Chunks:")
for doc in retrieved_docs:
    print("\n🔹", doc)
