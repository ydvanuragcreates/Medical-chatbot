# Corrected code for app.py

import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

# Import your custom helper functions and prompt
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# Import the correct LangChain components
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- App Initialization ---
app = Flask(__name__)

# --- Load Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Initialize Core Components ---
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot" # Make sure this is the correct name of your index

# Load the existing Pinecone index as a vector store
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Create the Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create the Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize the LLM (OpenAI)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.4,
    openai_api_key=OPENAI_API_KEY
)

# Create the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"User input: {input_text}")

        response = rag_chain.invoke({"input": input_text})
        print(f"Bot response: {response['answer']}")
        
        return str(response["answer"])
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

# --- Main Execution Block ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)