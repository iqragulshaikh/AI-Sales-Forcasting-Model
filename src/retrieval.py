import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key from .env file
load_dotenv()

# Path where the FAISS index will be saved on disk
VECTOR_STORE_PATH = "faiss_index"


def build_knowledge_base(docs_path):
    """
    Loads all .md documents from docs_path, splits them into chunks,
    creates embeddings, and stores them in a FAISS vector database.
    """
    # Load all markdown files from the given folder
    loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()

    # Split long documents into smaller chunks (easier to search accurately)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # roughly 500 characters per chunk
        chunk_overlap=50     # small overlap so context isn't cut off awkwardly
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using Gemini's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Build the FAISS vector store from the chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save it to disk so we don't have to rebuild it every time
    vectorstore.save_local(VECTOR_STORE_PATH)

    print(f"Knowledge base built with {len(chunks)} chunks from {len(documents)} documents.")
    return vectorstore


def retrieve_context(question, k=3):
    """
    Loads the saved FAISS vector store and retrieves the most relevant
    chunks for a given question. Returns a list of chunks with source info.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Load the previously saved vector store
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # safe here since we created this file ourselves
    )

    # Search for the top k most relevant chunks
    results = vectorstore.similarity_search(question, k=k)

    # Return chunks along with their source file for transparency
    retrieved = []
    for doc in results:
        retrieved.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown")
        })

    return retrieved


# Quick manual test — only runs if you execute this file directly
if __name__ == "__main__":
    print("Building knowledge base from docs/ folder...")
    build_knowledge_base("docs")

    print("\nTesting retrieval...")
    test_question = "What are the model's limitations?"
    context = retrieve_context(test_question)

    for i, chunk in enumerate(context):
        print(f"\n--- Chunk {i+1} (from {chunk['source']}) ---")
        print(chunk["content"][:200])  # print first 200 characters