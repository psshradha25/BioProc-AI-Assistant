from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# LOAD PDF
# =====================================================

print("\nLoading PDF...\n")

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

print(f"PDF Loaded Successfully")
print(f"Total Pages Loaded: {len(documents)}")

# =====================================================
# SPLIT DOCUMENT
# =====================================================

print("\nSplitting document into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"Total Chunks Created: {len(docs)}")

# =====================================================
# EMBEDDINGS
# =====================================================

print("\nLoading embedding model...\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")

# =====================================================
# CREATE VECTOR DATABASE
# =====================================================

print("\nCreating vector database...\n")

db = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="./chroma_db"
)

print("Vector DB Created Successfully")

# =====================================================
# TEST SEARCH
# =====================================================

print("\nTesting semantic search...\n")

query = "Protein A elution pH"

results = db.similarity_search(
    query,
    k=2
)

print(f"\nTop Results for Query: {query}\n")

for i, result in enumerate(results):

    print(f"\nResult {i+1}:\n")

    print(result.page_content)

    print("\n" + "="*50)