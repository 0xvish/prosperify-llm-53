from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load PDF
loader = PyPDFLoader("../intro_to_finance.pdf")
docs = loader.load()

# Load Embedding Model with GPU
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# Store in ChromaDB
db = Chroma.from_documents(docs, embedding=embed_model, persist_directory="./chroma_db")
db.persist()

print("✅ Data stored in ChromaDB!")
