# assignment_cybernetyx
Assignment: As a take-home task, we propose implementing a lightweight FastAPI server for RAG. This server should utilize ChromaDB’s persistent client for ingesting and querying documents (PDF, DOC, DOCX, TXT). Leverage sentence-transformers/all-MiniLM-L6-v2 (CPU) from Hugging Face for embeddings. Ensure non-blocking API endpoints and efficient concurrency handling.

here's my solution; done on google colab :
![Screenshot 2024-11-07 080535](https://github.com/user-attachments/assets/629e94e1-617e-4228-a94c-0aff1857d225)
1. # setting up the environment :
!pip install fastapi uvicorn chromadb sentence-transformers pyngrok
!pip install python-docx fitz  # For handling DOCX and PDF files

2. # start up using ngrok for tunneling purposes :
from pyngrok import ngrok
NGROK_AUTH_TOKEN = "2oV9by8KYxBeKJx6c1yqTuzAM0m_JiCgCtAnaidwVJHRFDbC"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
# Start an HTTP tunnel on the default FastAPI port 8000
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

![Screenshot 2024-11-07 081339](https://github.com/user-attachments/assets/3e41c1b3-0db8-4f27-87a2-e12ddffd1ae4)

3. # implimenting the fastAPI code :
# 3.1. Import Necessary Libraries and Configure ChromaDB and Embedding Model
!pip install chromadb # install chromadb
!pip uninstall fitz -y
!pip install pymupdf
from fastapi import FastAPI, UploadFile
from sentence_transformers import SentenceTransformer
import asyncio
import fitz  # For PDF processing
from docx import Document
import uvicorn
import chromadb # Import the main chromadb package
app = FastAPI()
# Initialize the embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
# Create a Chroma client
from chromadb.config import Settings
chroma_client = chromadb.Client(Settings(persist_directory="/content/chroma_storage")) # Use chromadb.Client to create the client

![image](https://github.com/user-attachments/assets/f45e747c-e8b9-4a82-aef1-60f323b4e5e3)

#3. 2. defining utilities for file processing :
async def extract_text(file: UploadFile):
    # Handle different file types
    if file.filename.endswith(".pdf"):
        with fitz.open(stream=await file.read(), filetype="pdf") as pdf:
            text = "".join([page.get_text() for page in pdf])
    elif file.filename.endswith(".docx"):
        doc = Document(await file.read())
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.filename.endswith(".txt"):
        text = (await file.read()).decode("utf-8")
    return text

![Screenshot 2024-11-07 081432](https://github.com/user-attachments/assets/21efa319-9332-468f-bbe9-3c9a91f279c0)


# 4.Create FastAPI Endpoints for Document Ingestion and Querying:
!pip install python-multipart 
from fastapi import FastAPI, UploadFile, Form, File 
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    text = await extract_text(file)
    embeddings = embedding_model.encode(text)
    chroma_client.insert("documents", embeddings, metadata={"filename": file.filename})
    return {"status": "Document ingested successfully"}

@app.get("/query")
async def query_documents(query: str):
    query_embedding = embedding_model.encode(query)
    results = chroma_client.query("documents", query_embedding)
    return {"results": results}

![image](https://github.com/user-attachments/assets/78cc2767-095c-466d-9e4f-021be24c8b99)


# 5. run fastAPI using uvicorn :
import nest_asyncio
import uvicorn

# Allow nested event loops (required for Colab):
# Since Colab doesn’t support running Uvicorn directly as a standalone server, we can launch it inside an asynchronous cell.
nest_asyncio.apply()

# Run Uvicorn server
uvicorn.run(app, host="0.0.0.0", port=8000)

