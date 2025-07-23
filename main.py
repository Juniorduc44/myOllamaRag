import os
import logging
import tracemalloc
import gc
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import chromadb
import fitz  # PyMuPDF
from test_ollama_server_status import ensure_ollama_server_running
import json
import subprocess
import sys

# Enable tracemalloc to trace memory allocations
tracemalloc.start()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment
MODEL = os.getenv("MODEL", "deepseek-r1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", f"http://localhost:{OLLAMA_PORT}")
CHROMA_PATH = os.getenv("CHROMA_PATH", ".praison")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "praison")
GPU_MEMORY_PERCENTAGE = int(os.getenv("GPU_MEMORY_PERCENTAGE", "50"))

def check_gpu_status():
    """Check if GPU is available and being used by Ollama."""
    try:
        # Check NVIDIA GPU memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            gpu_name = gpu_info[0]
            memory_used = int(gpu_info[1])
            memory_total = int(gpu_info[2])
            memory_free = int(gpu_info[3])
            
            target_memory = int(memory_total * (GPU_MEMORY_PERCENTAGE / 100))
            usage_percentage = (memory_used / memory_total) * 100
            
            logger.info(f"🎮 GPU: {gpu_name}")
            logger.info(f"📊 Memory: {memory_used}MB/{memory_total}MB ({usage_percentage:.1f}%)")
            logger.info(f"🎯 Target usage: {target_memory}MB ({GPU_MEMORY_PERCENTAGE}%)")
            
            # Check if Ollama is using GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if 'ollama' in result.stdout.lower():
                logger.info("✅ Ollama is using GPU")
            else:
                logger.warning("⚠️  Ollama might not be using GPU")
            return True, memory_total, target_memory
        else:
            logger.warning("❌ No NVIDIA GPU detected")
            return False, 0, 0
    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
        return False, 0, 0

def validate_configuration():
    """Validate the Ollama configuration."""
    logger.info(f"✅ Using Ollama with model: {MODEL}")
    logger.info(f"✅ Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"✅ Ollama URL: {OLLAMA_URL}")
    logger.info(f"✅ Chroma path: {CHROMA_PATH}, collection: {CHROMA_COLLECTION}")
    logger.info(f"🎯 GPU memory target: {GPU_MEMORY_PERCENTAGE}%")
    
    # Check GPU status with memory info
    check_gpu_status()
    
    if "OPENAI_API_KEY" in os.environ:
        logger.warning("OPENAI_API_KEY found in environment; removing to ensure Ollama-only usage")
        del os.environ["OPENAI_API_KEY"]
    return True

def list_ollama_models(url):
    """List available models from the Ollama server."""
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
        response = session.get(f"{url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            logger.info(f"Available Ollama models: {models}")
            return models
        else:
            logger.error(f"Failed to list Ollama models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []

def get_ollama_embedding(text, model, url):
    """Generate embeddings for text using Ollama's embeddings API."""
    try:
        # Truncate text if too long (Ollama has limits)
        max_length = 8192
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Truncated text for embedding from {len(text)} to {max_length} characters")
        
        payload = {"model": model, "prompt": text}
        response = requests.post(f"{url}/api/embeddings", json=payload, timeout=60)
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            if embedding:
                logger.info(f"Generated embedding for text (length: {len(text)})")
                return embedding
            logger.error("No embedding returned from Ollama")
            return None
        logger.error(f"Failed to generate embedding: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def query_ollama_streaming(prompt, model, url, context="", temperature=0.1):
    """Query the Ollama model with streaming support for real-time responses."""
    try:
        # Limit context size to prevent issues
        max_context_length = 3000
        if context and len(context) > max_context_length:
            context = context[:max_context_length] + "\n[...context truncated...]"
            logger.warning(f"Truncated context from {len(context)} to {max_context_length} characters")
        
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:" if context else prompt
        
        # Use streaming for real-time responses
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
                "num_ctx": 4096,  # Context window size
                "num_gpu": 99,  # Use all available GPU layers
                "stop": ["Question:", "Context:", "\n\n---"]
            }
        }
        
        response = requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=300)
        
        if response.status_code == 200:
            full_response = ""
            print("🤖 Assistant: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        token = json_response.get("response", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response += token
                        
                        # Check if generation is done
                        if json_response.get("done", False):
                            print()  # New line after response
                            logger.info("Streaming query completed successfully")
                            break
                    except json.JSONDecodeError:
                        continue
            
            return full_response.strip()
        else:
            logger.error(f"Failed to query Ollama: {response.status_code} - {response.text}")
            return None
    except requests.Timeout:
        logger.error("Ollama query timed out even with streaming. Check GPU usage.")
        return "Sorry, the query timed out. Please check if Ollama is properly using your GPU."
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better embedding coverage."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < text_length else text_length
    
    return chunks

def setup_chroma_collection(text_files, collection_name, chroma_path, embedding_model, ollama_url):
    """Setup Chroma vector store with embeddings from text files."""
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        
        # Delete existing collection to start fresh
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        collection = client.create_collection(name=collection_name)
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        doc_id = 0
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if not text.strip():
                    logger.warning(f"Empty content in {file_path}; skipping")
                    continue
                
                # Chunk the text for better retrieval
                chunks = chunk_text(text, chunk_size=1200, overlap=200)
                logger.info(f"Split {file_path} into {len(chunks)} chunks")
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Generate embedding for each chunk
                    embedding = get_ollama_embedding(chunk, embedding_model, ollama_url)
                    if embedding:
                        documents.append(chunk)
                        embeddings.append(embedding)
                        metadatas.append({
                            "source": os.path.basename(file_path),
                            "chunk": chunk_idx,
                            "total_chunks": len(chunks)
                        })
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1
                    else:
                        logger.warning(f"Failed to generate embedding for chunk {chunk_idx} of {file_path}")
                
                logger.info(f"Added {len(chunks)} chunks from {file_path} to collection")
                
            except Exception as e:
                logger.error(f"Error processing {file_path} for Chroma: {e}")
        
        if documents and embeddings:
            # Add in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                collection.add(
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            logger.info(f"Added {len(documents)} document chunks to Chroma collection")
        else:
            logger.warning("No documents added to Chroma collection")
        
        return collection
    except Exception as e:
        logger.error(f"Error setting up Chroma collection: {e}")
        import traceback
        traceback.print_exc()
        return None

def query_knowledge_base(query, collection, embedding_model, ollama_url, top_k=3):
    """Query the Chroma vector store for relevant documents."""
    try:
        query_embedding = get_ollama_embedding(query, embedding_model, ollama_url)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Get the actual count of documents in collection
        count = collection.count()
        actual_k = min(top_k, count)
        
        if actual_k == 0:
            logger.warning("No documents in collection")
            return []
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k
        )
        
        contexts = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            context_info = f"[Source: {metadata['source']}, Chunk {metadata['chunk']+1}/{metadata['total_chunks']}]\n{doc}"
            contexts.append(context_info)
        
        logger.info(f"Retrieved {len(contexts)} relevant document chunks from Chroma")
        return contexts
    except Exception as e:
        logger.error(f"Error querying Chroma: {e}")
        import traceback
        traceback.print_exc()
        return []

def find_knowledge_files():
    """Find supported knowledge files in the documents folder."""
    knowledge_files = []
    supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
    documents_dir = "./documents"
    
    if not os.path.exists(documents_dir):
        logger.error(f"Documents directory '{documents_dir}' does not exist!")
        logger.info("Please create a 'documents' folder and place your knowledge files there.")
        return knowledge_files
    
    if not os.path.isdir(documents_dir):
        logger.error(f"'{documents_dir}' exists but is not a directory!")
        return knowledge_files
    
    for root, _, files in os.walk(documents_dir):
        for filename in files:
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                knowledge_files.append(filepath)
                relative_path = os.path.relpath(filepath, documents_dir)
                logger.info(f"Found knowledge file: {relative_path} ({ext})")
    
    return knowledge_files

def verify_knowledge_sources():
    """Verify that knowledge sources are available and valid."""
    knowledge_files = find_knowledge_files()
    if not knowledge_files:
        logger.error("No knowledge files found in the documents folder!")
        logger.info("Supported file types: .pdf, .txt, .md, .doc, .docx")
        logger.info("Create the documents folder and place your files there: mkdir documents")
        return None
    
    valid_files = []
    for file in knowledge_files:
        try:
            file_size = os.path.getsize(file)
            if file_size == 0:
                logger.warning(f"File {file} is empty and will be skipped")
                continue
            relative_path = os.path.relpath(file, "./documents")
            logger.info(f"  - {relative_path} ({file_size:,} bytes)")
            valid_files.append(file)
        except Exception as e:
            logger.warning(f"Error accessing file {file}: {e}")
    
    if not valid_files:
        logger.error("No valid knowledge files found after verification!")
        return None
    
    logger.info(f"Found {len(valid_files)} valid knowledge file(s) in documents folder")
    return valid_files

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                text_content += f"\n--- Page {page_num + 1} ---\n{text}"
        doc.close()
        
        if not text_content.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return None
        
        logger.info(f"Successfully extracted text from {pdf_path} using PyMuPDF")
        return text_content.strip()
    except ImportError:
        logger.error("PyMuPDF not installed! Please install it: pip install PyMuPDF")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def create_text_files_from_pdfs(knowledge_files):
    """Convert PDFs to text files for processing."""
    text_files = []
    temp_dir = "./temp_extracted"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in knowledge_files:
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            text_content = extract_text_from_pdf(file_path)
            if text_content:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                text_file_path = os.path.join(temp_dir, f"{base_name}_extracted.txt")
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Content extracted from: {os.path.relpath(file_path, './documents')}\n\n")
                    f.write(text_content)
                text_files.append(text_file_path)
                logger.info(f"Created text file: {text_file_path} ({len(text_content)} characters)")
            else:
                logger.warning(f"Could not extract text from {file_path}")
        else:
            text_files.append(file_path)
    
    return text_files, temp_dir

def safe_close_sockets():
    """Safely close any open sockets."""
    try:
        closed_count = 0
        for obj in gc.get_objects():
            if isinstance(obj, socket.socket) and hasattr(obj, 'fileno') and obj.fileno() != -1:
                try:
                    obj.close()
                    closed_count += 1
                except (OSError, AttributeError):
                    pass
        if closed_count > 0:
            logger.info(f"Closed {closed_count} open sockets")
    except Exception as e:
        logger.warning(f"Error during socket cleanup: {e}")

def optimize_ollama_for_gpu():
    """Ensure Ollama is optimized for GPU usage."""
    try:
        # Set environment variables for GPU optimization
        os.environ['OLLAMA_NUM_GPU'] = '99'  # Use all available GPU layers
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
        
        logger.info("🎮 GPU optimization settings applied")
        
        # Check if model needs to be reloaded for GPU
        # This is done by making a simple query to "warm up" the model
        logger.info("🔥 Warming up model on GPU...")
        payload = {
            "model": MODEL,
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 1, "num_gpu": 99}
        }
        requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        logger.info("✅ Model warmed up on GPU")
    except Exception as e:
        logger.warning(f"Could not optimize for GPU: {e}")

def main():
    server_process = None
    temp_dir = None
    collection = None
    
    try:
        logger.info("Starting Ollama RAG Tool...")
        
        # Validate configuration
        if not validate_configuration():
            logger.error("❌ Configuration validation failed!")
            return
        
        # Find and verify knowledge sources
        knowledge_files = verify_knowledge_sources()
        if not knowledge_files:
            return
        
        # Convert PDFs to text files
        processed_files, temp_dir = create_text_files_from_pdfs(knowledge_files)
        if not processed_files:
            logger.error("No usable knowledge files after processing")
            return
        
        # Ensure Ollama server is running
        logger.info(f"Ensuring Ollama server is running on {OLLAMA_URL}")
        server_process = ensure_ollama_server_running(OLLAMA_URL, OLLAMA_PORT)
        
        # Optimize for GPU
        optimize_ollama_for_gpu()
        
        # Check for required models
        available_models = list_ollama_models(OLLAMA_URL)
        if MODEL not in available_models:
            logger.error(f"Model {MODEL} not found!")
            logger.info(f"Please pull the model: ollama pull {MODEL}")
            return
        if EMBEDDING_MODEL not in available_models:
            logger.error(f"Embedding model {EMBEDDING_MODEL} not found!")
            logger.info(f"Please pull the model: ollama pull {EMBEDDING_MODEL}")
            return
        
        # Setup Chroma vector store
        logger.info("Setting up Chroma vector store with document chunks...")
        collection = setup_chroma_collection(
            processed_files, 
            CHROMA_COLLECTION, 
            CHROMA_PATH, 
            EMBEDDING_MODEL, 
            OLLAMA_URL
        )
        if not collection:
            logger.error("Failed to setup Chroma collection")
            return
        
        # Interactive chat loop
        logger.info("RAG Agent is ready! Type 'quit', 'exit', or 'q' to stop.")
        print("\n" + "="*60)
        print("🤖 Ollama RAG Tool Ready!")
        print(f"🔗 Provider: Ollama ({OLLAMA_URL})")
        print(f"🎮 GPU: RTX 4090 (ensure CUDA is active)")
        print(f"📚 Knowledge Sources: {len(knowledge_files)} file(s)")
        for f in knowledge_files:
            relative_path = os.path.relpath(f, "./documents")
            print(f"   📄 {relative_path}")
        print(f"🧠 Model: {MODEL}")
        print(f"🧠 Embedding Model: {EMBEDDING_MODEL}")
        print(f"📊 Total chunks in knowledge base: {collection.count()}")
        print("="*60)
        print("\n💡 Using streaming for real-time responses...")
        print("💡 To verify GPU usage, run: watch -n 1 nvidia-smi")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    continue
                
                # Query knowledge base
                print("🔍 Searching knowledge base...")
                contexts = query_knowledge_base(user_input, collection, EMBEDDING_MODEL, OLLAMA_URL, top_k=3)
                
                if contexts:
                    context = "\n\n---\n\n".join(contexts)
                    print(f"📄 Found {len(contexts)} relevant context(s)")
                else:
                    context = ""
                    print("📄 No relevant context found, answering based on general knowledge")
                
                # Query Ollama with streaming
                response = query_ollama_streaming(
                    user_input,
                    MODEL,
                    OLLAMA_URL,
                    context=context,
                    temperature=0.1
                )
                
                if not response:
                    print("❌ Failed to get a response. Check GPU status with: nvidia-smi")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"❌ Error: {e}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Cleaning up resources...")
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
        safe_close_sockets()
        if server_process:
            try:
                server_process.terminate()
                logger.info("Ollama server process terminated")
            except Exception as e:
                logger.error(f"Error terminating server process: {e}")

if __name__ == "__main__":
    main()
