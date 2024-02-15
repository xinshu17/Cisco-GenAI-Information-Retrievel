import tempfile
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import base64
import json


from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import pandas as pd

import streamlit as st

# Class to store the documents (better fit with further retrival methods)
class TesseractDocument:
    """
    A document that has been processed by Tesseract.
    """
    def __init__(self, source, page_content):
        self.metadata = {'source': source}
        self.page_content = page_content
        
# Temporary directory to store the uploaded PDFs
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
# Temporary directory to store the splitted embeddings
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="Demo")
st.title("Cisco Document Intelligence Demo")

def load_documents():
    """
    Load the documents from the temp directory.
    """
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def load_documents_tesseract():
    """
    Load the documents from the temp directory and extract text from them using Tesseract.
    
    """
    documents = []

    # Iterate over each PDF in the temp directory
    for pdf_file in TMP_DIR.iterdir():
        if pdf_file.suffix == ".pdf":  # Ensure that it's a PDF file
            # Convert the PDF to a list of images (one per page)
            images = convert_from_path(pdf_file.as_posix())  # Explicitly convert the path to a string

            # Extract text from each image
            for image in images:
                text = pytesseract.image_to_string(image)
                doc = TesseractDocument(source=pdf_file.name, page_content=text)
                documents.append(doc)

    return documents

def create_download_link(df, title="Download CSV file", filename="data.csv"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
    return href

def split_documents(documents):
    """
    Split the documents into chunks of text.
    documents: list of Document objects
    """
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    """
    Create a retriever using the local vector store.
    texts: list of strings
    """
    vectordb = Chroma.from_documents(
        texts,
        embedding=OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key),
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

from langchain.prompts import PromptTemplate


def query_llm(retriever, query, selected_primer="You are a helpful assistant. Extract the information from this document. Say you do not know if you do not know."):
    '''
    Function to query the LLM.
    retriever: the retriever object
    query: the user input query
    selected_primer: the selected system prompt
    '''
    # Default system prompt
    system_prompt = selected_primer
    # Prepend the system prompt to the query
    modified_query = f"{system_prompt} {query}"
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=st.session_state.openai_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': modified_query, 'chat_history': st.session_state.messages})
    answer = result['answer']
    source_documents = result['source_documents']
    st.session_state.messages.append((query, answer, source_documents))
    return answer, source_documents


def input_fields():
    '''
    Function to create the input fields.
    '''
    with st.sidebar:
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")


    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    
    # Primer Selection
    primer_options = {
        "helpful": "You are a helpful assistant. Say you do not know if you do not know the answer.",
        "concise": "You are a concise and to-the-point assistant. Say you do not know if you do not know the answer.",
        "casual": "You are a friendly and assistant. Say you do not know if you do not know the answer."
    }
    selected_primer = st.sidebar.selectbox("Choose the assistant's style: (developing)", list(primer_options.keys()))
    # st.session_state.default_message = primer_options[selected_primer]

def extract_attributes_from_documents(retriever, attributes):
    """
    Extract the attributes from the documents.
    retriever: the retriever object
    attributes: list of attributes
    """
    results = []
    for attr in attributes:
        query = f"What is the {attr}?"
        response, _ = query_llm(retriever, query)
        results.append({"Attribute": attr, "Value": response})
    return results





# def extract_attributes_from_documents(retriever):
#     attributes_list = ["Name", "Product Type", "Dimensions", "Orientation", "Current Rating", "Voltage", "Frequency", "Impedance", "Capacitance", "Temperature"]
#     query = "Can you provide the " + ", ".join(attributes_list[:-1]) + ", and " + attributes_list[-1] + " from the document?"
#     response, _ = query_llm(retriever, query)
    
#     # TODO: Add parsing logic here to split the response into individual attributes
#     # For simplicity, let's assume the response format is "Name: XYZ, Product Type: ABC, ..."
#     attribute_values = {attr: val.strip() for attr, val in [item.split(":") for item in response.split(",")]}
    
#     results = [{"Attribute": attr, "Value": attribute_values.get(attr, "Not found")} for attr in attributes_list]
#     return results

import json
import os
import re
def save_json(df, filename, directory):
    """
    Save a DataFrame as a JSON file in a specified directory.

    :param df: pandas.DataFrame - DataFrame to save
    :param filename: str - the name of the file
    :param directory: str - the directory where to save the file
    """
    # Ensures that the directory exists
    os.makedirs(directory, exist_ok=True)
    # Regular expression to match the file name without path and extensions
    match = re.search(r'([^\\/]+?)(?:\.[^\.]+)?$', filename)
    # The table name is the first captured group of the regex
    table_name = match.group(1) if match else None

    # Sets the full path to save the file
    full_path = os.path.join(directory, f"{table_name}.json")
    print(full_path)
    # Convert DataFrame to JSON and save it
    df.to_json(full_path, orient='records', lines=True, force_ascii=False)


def clear_directories():
    '''
    Function to clear the temp and vector store directories.
    '''
    # Clearing the temp directory
    for item in TMP_DIR.iterdir():
        if item.is_file():
            item.unlink()
    
    # Clearing the vector store directory
    for item in LOCAL_VECTOR_STORE_DIR.iterdir():
        if item.is_file():
            item.unlink()
            
def clear_messages():
    '''
    Function to clear the messages.
    '''
    st.session_state.messages = []


def process_documents():
    """
    Function to process the documents.
    """
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                # Capture the uploaded file name
                pdf_name = source_doc.name
                
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                documents = load_documents_tesseract()

                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()

                texts = split_documents(documents)
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
                
                attributes_list = ["Name", "Product Type", "Dimensions", "Orientation", "Current Rating", "Voltage", "Frequency", "Impedance", "Capacitance", "Temperature"]
                # attributes_list = ["Name", "Product Type", "Temperature"]

                results = extract_attributes_from_documents(st.session_state.retriever, attributes_list)
                
                df = pd.DataFrame(results)

                # Use the captured PDF name as the key for the session state's tables
                if 'tables' not in st.session_state:
                    st.session_state.tables = {}
                st.session_state.tables[pdf_name] = df

        except Exception as e:
            st.error(f"An error occurred: {e}")
import os

def load_files_from_directory(directory_path):
    pdf_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            # Open the file and append the file object to the list
            pdf_files.append(open(os.path.join(directory_path, filename), 'rb'))
    return pdf_files


def boot():
    input_fields()
    
   # Text input for directory path
    dir_path = st.text_input("Enter directory path for PDFs:")

    # Button to load PDFs from the specified directory
    if st.button("Load PDFs from Directory"):
        if dir_path:
            try:
                # Use load_files_from_directory to load all PDFs
                pdf_files = load_files_from_directory(dir_path)
                st.session_state['source_docs'] = pdf_files  # Storing the file-like objects in session state
                st.success(f"Loaded {len(pdf_files)} PDF(s) from the directory.")
                # Process the loaded PDFs immediately
                process_documents()
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a directory path.")
            
    st.button("Submit Documents", on_click=process_documents)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display all stored tables in session state
    if 'tables' in st.session_state:
        for table_name, table_df in st.session_state.tables.items():
            with st.expander(f"Results from {table_name}"):  # This will display the PDF name
                st.write(table_df)
                json_filename = f"results_{table_name}.json"
                save_json(table_df, json_filename, "output/")  # This will save the JSON in the specified directory
                # Add a download button for each table
                st.markdown(create_download_link(table_df, title=f"Download results from {table_name} as CSV", filename=f"results_{table_name}.csv"), unsafe_allow_html=True)
            
    for message in st.session_state.messages:
        # Assuming each message is a tuple of the format (role, content, [optional source documents])
        role, content, *source_documents = message

        if role == 'human':
            st.chat_message('human').write(content)
        elif role == 'ai':
            st.chat_message('ai').write(content)
            if source_documents:  # if there are source documents associated with the message
                for doc_idx, doc in enumerate(source_documents[0], 1):  # source_documents[0] because *source_documents creates a list
                    with st.expander(f"Reference Document {doc_idx}"):
                        st.write(f"Source: {doc.metadata['source']}")
                        st.write(f"Content: {doc.page_content}")

    if query := st.chat_input():
        response, source_documents = query_llm(st.session_state.retriever, query)
        
        # Add the human's message to the session state
        st.session_state.messages.append(('human', query))
        st.chat_message("human").write(query)
        
        # Add the AI's response to the session state
        st.session_state.messages.append(('ai', response, source_documents))
        st.chat_message("ai").write(response)
        
        for doc_idx, doc in enumerate(source_documents, 1):
            with st.expander(f"Reference Document {doc_idx}"):
                st.write(f"Source: {doc.metadata['source']}")
                st.write(f"Content: {doc.page_content}")
                
    if st.button("Clear Memory"):
        clear_directories()
        clear_messages()
        st.success("Memory cleared!")


if __name__ == '__main__':
    boot()