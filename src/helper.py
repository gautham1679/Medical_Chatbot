# ✅ Updated imports for LangChain 0.2+

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Typing & schema
from typing import List
from langchain_core.documents import Document




def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )


    documents=loader.load()
    return documents



def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    minimal_docs : List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document( 
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk
   

def download_embeddings():
    model_name="BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        

    )
    return embeddings

embedding=download_embeddings()