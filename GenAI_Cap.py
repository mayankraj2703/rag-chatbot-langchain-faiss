# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 04:32:55 2026

@author: user
"""

# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_classic.chains import ConversationalRetrievalChain


# Load environment variables from .env
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.getenv(
    "DOCS_PATH",
    os.path.join(BASE_DIR, "documents")
)


def get_llm(docs_path=DOCS_PATH):

    # Get API key securely from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please add it to your .env file."
        )

    # Make sure documents directory exists
    os.makedirs(docs_path, exist_ok=True)

    # Load PDFs
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No PDF documents found in: {docs_path}"
        )

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key
    )

    # Create FAISS vector database
    vector_db = FAISS.from_documents(
        texts,
        embeddings
    )

    # Create LLM
    llm = OpenAI(
        api_key=openai_api_key
    )

    # Create conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )

    return qa