from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Initialize Flask App
app = Flask(__name__)

# Enable CORS (Allows Frontend to Communicate with Backend)
from flask_cors import CORS
CORS(app)

# Load Embedding Model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load ChromaDB
db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

# Initialize Gemini API (Replace with your API Key)
genai.configure(api_key="AIzaSyBovtxlggNMyru3W7ubKZOf7oyqkDtoJkA")

# Function to Retrieve Relevant Chunks
def get_relevant_docs(query, k=3):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Function to Get AI Response
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't find an answer."

@app.route("/")
def home():
    return "Prosperify Chatbot API is running!"

# API Endpoint for Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve relevant context from ChromaDB
    context = get_relevant_docs(user_query)
    full_prompt = f"""You are a highly experienced financial advisor in india dealing with only indian ruppee with deep knowledge of finance, investments, and wealth management. 
Your job is to provide well-explained and beginner-friendly responses along with teaching the users everything about the asked topic.
The users will be people from middle class house holds so keep that in mind while explaining to them and they often struggle to understand complex finance terminologies.

You have to guide them in the areas of finance and also help them in budgeting, investment planning, debt management, retriement planning, student schooling financial planning and other financial ares.

Use the following financial information to answer the user's query. 
If any financial terms appear, explain them in simple words before answering to ensure the user fully understands. 
Also, provide practical insights or examples where applicable.

Retrieved Financial Information:
{context}

User's Question:
{user_query}

even if there is no financial information retrieved(context), then use the user_query to understand the question and help them by doing what the user has asked.
Provide a structured, clear, and detailed response.
Only if the user provides query in hindi or asks in the prompt to give the response in hindi, translate it to english, generate response, convert it to hindi and send it."""

    # Get response from Gemini
    bot_response = get_gemini_response(full_prompt)

    return jsonify({"response": bot_response})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
