from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM as Ollama  # ✅ Updated modern Ollama import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__)

# ✅ Initialize Hugging Face embeddings
embeddings = download_embeddings()

# ✅ Connect to Pinecone index
index_name = "medical-chatbot"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ✅ Initialize local Ollama model (running on your PC)
# Make sure "ollama serve" is running and "mistral" is downloaded via "ollama pull mistral"
llm = Ollama(model="mistral", temperature=0.2)

# ✅ Create retriever (to fetch relevant context from Pinecone)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ Function to combine multiple retrieved documents into one text block
def combine_docs(retrieved_dict):
    docs = retrieved_dict["documents"] if isinstance(retrieved_dict, dict) else retrieved_dict
    return "\n\n".join([d.page_content for d in docs])

# ✅ Define the system prompt for MediBot
prompt = ChatPromptTemplate.from_template(
    """
    You are MediBot — a knowledgeable and responsible AI medical assistant.
    Use the retrieved medical context below to answer the user's question accurately,
    clearly, and concisely. If the answer is unclear, say:
    "I'm not certain — please consult a medical professional."

    Context:
    {context}

    Question:
    {input}
    """
)

# ✅ Build the modern LangChain RAG pipeline
rag_chain = (
    {
        "context": retriever | combine_docs,  # Get info from Pinecone
        "input": RunnablePassthrough()        # Pass user input directly
    }
    | prompt
    | llm
    | StrOutputParser()                      # Parse Ollama output into plain text
)

# ✅ Flask routes
@app.route("/")
def index():
    return render_template("chat.html")  # Your frontend chat UI

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User: {msg}")

    try:
        # 🔍 Retrieve relevant context from Pinecone
        docs = retriever.invoke(msg)
        print("\n🔍 Retrieved from Pinecone:")
        for d in docs:
            print("-", d.metadata.get("source"), "→", d.page_content[:100], "...\n")

        # 💬 Generate response using Ollama + retrieved context
        response = rag_chain.invoke({"input": msg})
        print("Response:", response)

        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
