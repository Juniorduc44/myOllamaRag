First intall dependencies
 - requirements
 use the script pip install requirements.txt



Here's the README.md file ready to be copied directly into your GitHub repository:

markdown
# Ollama RAG Project

Welcome to the Ollama RAG (Retrieval-Augmented Generation) Project! This repository contains the source code and setup instructions for running a local RAG system using [Ollama](https://ollama.ai/), which is an open-source tool for running large language models on your machine.

## Overview

This project demonstrates how to set up and use a Retrieval-Augmented Generation system with Ollama. RAG combines the power of language models with external knowledge retrieval to provide more accurate and context-aware responses. Here, we leverage Ollama to run models locally, ensuring privacy and control over data.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **Ollama**: Follow the installation guide on the [Ollama website](https://ollama.ai/)
- **Git**: For cloning this repository



 # Point to your local Ollama server
```bash 
export OPENAI_BASE_URL=http://localhost:11434/v1
```
# Ollama doesn't require a real OpenAI key, so "fake-key" works
```bash
export OPENAI_API_KEY=fake-key
```