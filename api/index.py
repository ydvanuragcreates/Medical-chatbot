import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize Core Components
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           temperature=0.4,
                           google_api_key=GOOGLE_API_KEY)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# The 'app' variable is what Vercel will look for
app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(f"User input: {input_text}")

    response = rag_chain.invoke({"input": input_text})
    print(f"Bot response: {response['answer']}")
    
    return str(response["answer"])