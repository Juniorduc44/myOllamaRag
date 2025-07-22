import requests
import socket
import subprocess
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    """Check if a port is in use by attempting to bind to it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except socket.error:
            return True

def check_ollama_server(url):
    """Check if the Ollama server is running by querying its API."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info(f"Ollama server is running at {url}")
            return True
        logger.warning(f"Ollama server responded with status {response.status_code} at {url}")
        return False
    except requests.ConnectionError:
        logger.error(f"Failed to connect to Ollama server at {url}")
        return False

def free_port(port, force=False):
    """Optionally free a port by terminating the process using it."""
    if not force:
        logger.info(f"Port {port} is in use, but force=False; skipping termination")
        return False
    try:
        result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            pid = lines[1].split()[1]
            subprocess.run(["kill", "-9", pid], check=True)
            logger.info(f"Killed process {pid} using port {port}")
            return True
        logger.info(f"No process found using port {port}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error freeing port {port}: {e}")
        return False

def start_ollama_server(port):
    """Start the Ollama server on the specified port, handling conflicts."""
    try:
        if is_port_in_use(port):
            logger.warning(f"Port {port} is in use")
            if not free_port(port, force=False):  # Set force=True to auto-kill processes
                raise RuntimeError(f"Port {port} is in use and could not be freed")
            time.sleep(2)  # Wait for port to free up
        cmd = ["ollama", "serve"]
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Started Ollama server on port {port} with PID {process.pid}")
        time.sleep(5)  # Give the server time to start
        return process
    except Exception as e:
        logger.error(f"Failed to start Ollama server: {e}")
        raise

def ensure_ollama_server_running(url, port):
    """Ensure the Ollama server is running, starting it if necessary."""
    if not check_ollama_server(url):
        logger.info(f"Ollama server not running at {url}. Starting on port {port}...")
        process = start_ollama_server(port)
        if check_ollama_server(url):
            logger.info("Ollama server started successfully")
            return process
        raise RuntimeError(f"Failed to start Ollama server on port {port}")
    return None