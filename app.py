from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__)

# ✅ Initialize embeddings (Hugging Face)
embeddings = download_embeddings()

# ✅ Connect to Pinecone index
index_name = "medical-chatbot"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ✅ Initialize local LLM (Ollama)
llm = OllamaLLM(model="mistral", temperature=0.2)

# ✅ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ✅ Combine retrieved documents into one string
def combine_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# ✅ Ensure retriever always gets plain string input
def safe_query(x):
    """Ensure retriever only receives plain text."""
    if isinstance(x, dict):
        return str(x.get("input", ""))
    return str(x)


# ✅ Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are MediBot — a helpful, responsible medical assistant.
    Use the retrieved medical context below to answer the question safely, factually, and concisely.
    If you are not sure about the answer, say: "I'm not certain, please consult a medical professional."

    Context:
    {context}

    Question:
    {input}
    """
)

# ✅ RAG Chain (Retriever → Prompt → Ollama → Output Parser)
rag_chain = (
    {
        "context": safe_query | retriever | combine_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ✅ Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("🧑‍💻 User:", msg)

    try:
        # 🔍 Step 1: Retrieve documents
        docs = retriever.invoke(str(msg))
        print("\n🔍 Retrieved from Pinecone:")
        for d in docs:
            print("-", d.metadata.get("source"), "→", d.page_content[:120], "...\n")

        # 🤖 Step 2: Run RAG chain
        response = rag_chain.invoke({"input": str(msg)})
        print("🧠 MediBot Response:", response)

        return jsonify({"response": response})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)})


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
