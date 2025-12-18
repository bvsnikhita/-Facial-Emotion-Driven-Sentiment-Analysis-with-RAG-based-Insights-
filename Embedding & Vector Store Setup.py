from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

# Load synthetic reviews
df = pd.read_csv("synthetic_reviews.csv")

# Load embeddings
embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS database with LangChain
db = FAISS.from_texts(df["synthetic_review"].tolist(), embeddings)

# Save FAISS DB in a folder
db.save_local("faiss_index_folder")  # <-- saves in a folder, not a single file
print("✅ FAISS vector database created and saved in 'faiss_index_folder'")
