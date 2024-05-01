from fastapi import APIRouter,Form
from fastapi import HTTPException
import pickle
from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException
from tempfile import NamedTemporaryFile
from langchain_community.llms import AzureOpenAI
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.embeddings import AzureOpenAIEmbeddings,HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import certifi
import os
import openai
import nltk
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

router = APIRouter()

os.environ['TRANSFORMERS_CACHE'] = './temp'
os.environ['HF_HOME'] = './temp'

print(f"OpenAI API Key: {os.getenv('AZURE_OPENAI_KEY')}")
# Load Azure OpenAI credentials from .env file
load_dotenv()
modelPath = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja" #1536 Dimensionality same as collection in Chroma
 
        # Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}
 
        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}
 
        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
        )
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
nltk.download("averaged_perceptron_tagger")
deployment_name_mod = os.getenv("deployment_name")
os.environ["SSL_CERT_FILE"] = certifi.where()


TEMP_FOLDER = Path("temp")
TEMP_FOLDER.mkdir(exist_ok=True)  # Ensure the temp folder exists

# Define file handling and processing functions

async def convert_excel_to_csv(excel_file):
    df = pd.read_excel(excel_file.file, engine='openpyxl')
    temp_csv_path = TEMP_FOLDER / f"{excel_file.filename}.csv"
    df.to_csv(temp_csv_path, index=False, encoding='utf-8')
    return temp_csv_path

async def save_uploaded_file(uploaded_file, suffix):
    temp_file_path = TEMP_FOLDER / f"{uploaded_file.filename}{suffix}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.file.read())
    return temp_file_path

async def process_csv(file_path):
    loader = CSVLoader(str(file_path))  # Convert path to string
    data = loader.load()
    return data

async def process_powerpoint(file_path):
    loader = UnstructuredPowerPointLoader(str(file_path))  # Convert path to string
    data = loader.load()
    return data

async def process_file(uploaded_file: UploadFile = File(...)):
    # Determine the file type and process accordingly
    if uploaded_file.filename.endswith('.xlsx'):
        tmp_file_path = await convert_excel_to_csv(uploaded_file)
        data = await process_csv(tmp_file_path)
    elif uploaded_file.filename.endswith('.csv'):
        tmp_file_path = await save_uploaded_file(uploaded_file, "")
        data = await process_csv(tmp_file_path)
    elif uploaded_file.filename.endswith('.pptx'):
        tmp_file_path = await save_uploaded_file(uploaded_file, ".pptx")
        data = await process_powerpoint(tmp_file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Process the data to generate embeddings and a retrieval chain
    # embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002",api_key=azure_openai_key,api_version=api_version)
    embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
        )
    vectors = FAISS.from_documents(data, embeddings)
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=AzureChatOpenAI(api_key=openai.api_key ,api_version=openai.api_version, deployment_name=deployment_name_mod),
        retriever=vectors.as_retriever()
    )
    return chain

# Define your Azure OpenAI credentials
# azure_openai_key = 'YOUR_AZURE_OPENAI_KEY'
# azure_openai_endpoint = 'YOUR_AZURE_OPENAI_ENDPOINT'

@router.post("/process_pdf")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
):
    # Process the PDF and extract raw text
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Process the PDF and extract raw text
        raw_text = extract_text(temp_file_path)

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
        )

    # Create the document search index
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load the question-answering chain
    chain = load_qa_chain(AzureOpenAI(api_key=openai.api_key ,api_version=openai.api_version, deployment_name=deployment_name_mod), chain_type="stuff")
    
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    os.remove(temp_file_path)
    return {
        "query": query,
        "answer": response,
    }

@router.post("/process_excel/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = await process_file(file)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return {
        "query": query,
        "answer": response['answer'],
    }

@router.post("/process_csv/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = await process_file(file)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return {
        "query": query,
        "answer": response['answer'],
    }

@router.post("/process_ppt/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = await process_file(file)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return {
        "query": query,
        "answer": response['answer'],
    }

