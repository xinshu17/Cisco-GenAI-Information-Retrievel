# RAG Engine with Streamlit README

![Demo](data/demo.gif)

# Overview

This repository contains an integration of a Retrieval-Augmented Generation (RAG) engine with Streamlit. It enables users to extract information from Cisco PDF documents and interact with the data through a user-friendly Streamlit interface.

## Features

- **Streamlit Web App**: The project is built using Streamlit, providing an intuitive and interactive web interface for users.
- **Input Fields**: Users can input essential credentials like OpenAI API key.
- **Document Uploader**: Users can upload multiple PDF files, which are then processed for further analysis.
- **Document Splitting**: The uploaded PDFs are split into smaller text chunks, ensuring compatibility with models with token limits.
- **Vector Embeddings**: The text chunks are converted into vector embeddings, making it easier to perform retrieval and question-answering tasks.
- **Flexible Vector Storage**: You can choose to store vector embeddings either in Pinecone or a local vector store, providing flexibility and control.
- **Batch Process**: Running query in parallel to efficiently extract the information.
- **Interactive Conversations**: Users can engage in interactive conversations with the documents, asking questions and receiving answers. The chat history is preserved for reference.

## Prerequisites

Before running the project, make sure you have the following prerequisites:

* Python 3.7
* Required libraries from the `requirements.txt` file
* An active OpenAI API key
* PDF documents for analysis

## Steps for Running the model


1. **Environment Setup**: Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Credential Configuration**: Set up your `secrets.toml` configuration file:

   ```bash
   cd 02_Deliverables/gpt_streamlit  # Navigate to the credential directory
   mkdir -p .streamlit                # Ensure the .streamlit directory exists
   touch .streamlit/secrets.toml      # Create the secrets.toml file
   ```

   Populate `secrets.toml` with your OpenAI API key:

   ```toml
   openai_api_key = "YOUR_API_KEY_HERE"
   ```

Contributors

## Running the Web Application

Launch the Streamlit interface with the following command:

```bash
streamlit run rag_engine_cur.py
```


## Acknowledgments

Partial code referenced from: [Mir Abdullah Yaser](https://github.com/mirabdullahyaser) (https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit)
