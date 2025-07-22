from praisonaiagents import Agent
from dotenv import load_dotenv
import os
import logging
import tracemalloc
from test_ollama_server_status import ensure_ollama_server_running
import gc
import socket

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
PROVIDER = os.getenv("PROVIDER", "ollama").lower()
MODEL = os.getenv("MODEL", "deepseek-r1:8b")
ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
ollama_url = os.getenv("OLLAMA_BASE_URL", f"http://localhost:{ollama_port}")

def validate_configuration():
    """Validate the configuration based on the selected provider"""
    if PROVIDER == "ollama":
        logger.info(f"✅ Using OLLAMA provider with model: {MODEL}")
        logger.info(f"✅ Ollama URL: {ollama_url}")
        # Don't set OpenAI key for Ollama
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        return True
        
    elif PROVIDER == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_actual_openai_api_key_here":
            logger.error("❌ OpenAI provider selected but no valid API key found!")
            logger.error("❌ Please set your actual OpenAI API key in the .env file")
            return False
        logger.info(f"✅ Using OPENAI provider with model: {MODEL}")
        logger.info(f"✅ API Key: {openai_key[:8]}...")
        return True
    else:
        logger.error(f"❌ Invalid provider: {PROVIDER}. Must be 'ollama' or 'openai'")
        return False

def safe_close_sockets():
    """Safely close any open sockets without triggering proxy object initialization"""
    try:
        closed_count = 0
        for obj in gc.get_objects():
            # Only check actual socket objects, avoid proxy objects
            if (hasattr(obj, '__class__') and 
                obj.__class__.__module__ == 'socket' and 
                obj.__class__.__name__ == 'socket'):
                try:
                    if hasattr(obj, 'fileno') and obj.fileno() != -1:
                        obj.close()
                        closed_count += 1
                except (OSError, AttributeError):
                    # Socket already closed or invalid
                    pass
        if closed_count > 0:
            logger.info(f"Closed {closed_count} open sockets")
    except Exception as e:
        logger.warning(f"Error during socket cleanup: {e}")

def setup_agent_config():
    """Setup the agent configuration based on the selected provider"""
    if PROVIDER == "ollama":
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "praison",
                    "path": ".praison"
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": MODEL,
                    "temperature": 0,
                    "max_tokens": 8000,
                    "ollama_base_url": ollama_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest",
                    "ollama_base_url": ollama_url,
                    "embedding_dims": 1536
                }
            }
        }
    else:  # openai
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "praison",
                    "path": ".praison"
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MODEL,
                    "temperature": 0,
                    "max_tokens": 8000,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "embedding_dims": 1536
                }
            }
        }
    
    return config

def find_knowledge_files():
    """Find all knowledge files (PDFs, text files, etc.) in the documents folder"""
    knowledge_files = []
    supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
    
    documents_dir = "./documents"
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        logger.error(f"Documents directory '{documents_dir}' does not exist!")
        logger.info("Please create a 'documents' folder and place your knowledge files there.")
        return knowledge_files
    
    if not os.path.isdir(documents_dir):
        logger.error(f"'{documents_dir}' exists but is not a directory!")
        return knowledge_files
    
    # Recursively search the documents directory
    for root, dirs, files in os.walk(documents_dir):
        for filename in files:
            _, ext = os.path.splitext(filename.lower())
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                knowledge_files.append(filepath)
                relative_path = os.path.relpath(filepath, documents_dir)
                logger.info(f"Found knowledge file: {relative_path} ({ext})")
    
    return knowledge_files

def verify_knowledge_sources():
    """Verify that we have knowledge sources available in the documents folder"""
    knowledge_files = find_knowledge_files()
    
    if not knowledge_files:
        logger.error("No knowledge files found in the documents folder!")
        logger.info("Supported file types: .pdf, .txt, .md, .doc, .docx")
        logger.info("Please ensure you have at least one of these file types in the './documents/' directory")
        logger.info("Create the documents folder and place your files there: mkdir documents")
        return None
    
    logger.info(f"Found {len(knowledge_files)} knowledge file(s) in documents folder:")
    for file in knowledge_files:
        file_size = os.path.getsize(file)
        relative_path = os.path.relpath(file, "./documents")
        logger.info(f"  - {relative_path} ({file_size:,} bytes)")
    
    return knowledge_files

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using multiple methods for better accuracy"""
    text_content = ""
    
    try:
        # Try PyMuPDF first (most accurate)
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_content += page.get_text()
        doc.close()
        logger.info(f"Successfully extracted text from {pdf_path} using PyMuPDF")
        
    except ImportError:
        try:
            # Fallback to PyPDF2
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            logger.info(f"Successfully extracted text from {pdf_path} using PyPDF2")
            
        except ImportError:
            try:
                # Fallback to pdfplumber (best for tables)
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_content += text
                logger.info(f"Successfully extracted text from {pdf_path} using pdfplumber")
                
            except ImportError:
                logger.error("No PDF processing library found!")
                logger.error("Please install one of the following:")
                logger.error("  pip install PyMuPDF  # Recommended")
                logger.error("  pip install PyPDF2")
                logger.error("  pip install pdfplumber")
                return None
    
    return text_content.strip() if text_content.strip() else None

def create_text_files_from_pdfs(knowledge_files):
    """Convert PDFs to text files for better compatibility with the agent"""
    text_files = []
    temp_dir = "./temp_extracted"
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in knowledge_files:
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            # Extract text from PDF
            text_content = extract_text_from_pdf(file_path)
            if text_content:
                # Create a text file with the extracted content
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
            # Non-PDF files can be used directly
            text_files.append(file_path)
    
    return text_files, temp_dir

def main():
    server_process = None
    agent = None
    temp_dir = None
    
    try:
        logger.info("Starting Ollama RAG Tool...")
        
        # Validate configuration first
        if not validate_configuration():
            logger.error("❌ Configuration validation failed!")
            return
        
        # Find and verify knowledge sources
        knowledge_files = verify_knowledge_sources()
        if not knowledge_files:
            return
        
        # Convert PDFs to text files for better compatibility
        processed_files, temp_dir = create_text_files_from_pdfs(knowledge_files)
        if not processed_files:
            logger.error("No usable knowledge files after processing")
            return
        
        # Only ensure Ollama server is running if using Ollama
        if PROVIDER == "ollama":
            logger.info(f"Ensuring Ollama server is running on {ollama_url}")
            server_process = ensure_ollama_server_running(ollama_url, ollama_port)
        
        # Setup configuration
        config = setup_agent_config()
        logger.info(f"Using {PROVIDER.upper()} with model: {MODEL}")
        
        # Use the directory containing our processed files
        knowledge_source = temp_dir if temp_dir else "."
        
        # Initialize the Agent
        logger.info("Initializing Knowledge Agent...")
        agent = Agent(
            name="Knowledge Agent",
            instructions="You are a helpful assistant that answers questions based on the provided knowledge base. Be accurate and cite sources when possible. When referencing information, mention the source document name.",
            knowledge=[knowledge_source],
            knowledge_config=config,
            user_id="user1"
        )
        logger.info("Agent initialized successfully")
        
        # Interactive chat loop
        logger.info("RAG Agent is ready! Type 'quit', 'exit', or 'q' to stop.")
        print("\n" + "="*60)
        print("🤖 Ollama RAG Tool Ready!")
        print(f"🔗 Provider: {PROVIDER.upper()}")
        print(f"📚 Knowledge Sources: {len(knowledge_files)} file(s)")
        for f in knowledge_files:
            relative_path = os.path.relpath(f, "./documents")
            print(f"   📄 {relative_path}")
        print(f"🧠 Model: {MODEL}")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Assistant: ", end="")
                response = agent.chat(user_input)
                print(response)
                
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
        
        # Clean shutdown
        if agent:
            try:
                if hasattr(agent, "close"):
                    agent.close()
                    logger.info("Agent resources closed")
            except Exception as e:
                logger.error(f"Error closing agent: {e}")
        
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory: {e}")
        
        # Clean up sockets
        safe_close_sockets()
        
        # Terminate server process if we started it (only for Ollama)
        if server_process and PROVIDER == "ollama":
            try:
                server_process.terminate()
                logger.info("Ollama server process terminated")
            except Exception as e:
                logger.error(f"Error terminating server process: {e}")
        
        # Log memory stats
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logger.info("Top 5 memory allocations:")
            for stat in top_stats[:5]:
                logger.info(f"  {stat}")
        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")

if __name__ == "__main__":
    main()