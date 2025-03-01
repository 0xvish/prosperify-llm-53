from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Embedding Model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load ChromaDB
db = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

# Initialize Gemini API
genai.configure(api_key="AIzaSyBovtxlggNMyru3W7ubKZOf7oyqkDtoJkA")

# Store Chat Sessions
chat_sessions = {}  # { session_id: [{"role": "user/assistant", "content": "..."}] }

# Function to Retrieve Relevant Chunks
def get_relevant_docs(query, k=3):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# Function to Get AI Response
def get_gemini_response(conversation_history):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(conversation_history)
    return response.text if response else "Sorry, I couldn't find an answer."

@app.route("/")
def home():
    return "Prosperify Chatbot API is running!"

# API Endpoint to Start a New Chat
@app.route("/new_chat", methods=["POST"])
def new_chat():
    session_id = request.json.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID required"}), 400

    chat_sessions[session_id] = []  # Reset conversation history
    return jsonify({"message": "New chat started!"})

# API Endpoint for Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_query = data.get("query", "")

    if not session_id or not user_query:
        return jsonify({"error": "Session ID and Query are required"}), 400

    # Retrieve relevant context from ChromaDB
    context = get_relevant_docs(user_query)

    # Retrieve previous chat history
    chat_history = chat_sessions.get(session_id, [])

    # Construct prompt with conversation history
    full_prompt = f"""You are a highly experienced financial advisor in india dealing with only indian ruppee with deep knowledge of finance, investments, and wealth management. You are best at giving advises and making budget and investment plans for middle class households. 
Your job is to provide well-explained and beginner-friendly responses along with teaching the users everything about the asked topic.
The users will be people from middle class house holds so keep that in mind while explaining to them and they often struggle to understand complex finance terminologies.
You have to guide them in the areas of finance and also help them in budgeting, investment planning, debt management, retriement planning, student schooling financial planning and other financial ares.

Use this step-by-step process to ensure your script is first class:
step 1: Greet the customer warmly in different indian styles and answer the questions asked by user.
step 2: if the user has asked something to just learn then just explain it with simple language and 1-2 proper examples when needed.
step 3: if the user has entered some details for budgeting/ivestment/retirement planning or something else, carefully analyze the entered data and give a solid advice.
step 4: keep in mind the users preference of language(English/Hindi) and use that language to respond.
step 5: suggest the slightly risky plans and ideas too but give a note of risks wherever major risks are involved to the user.
step 6: give a note at the end only if you have suggested something risky.

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

    # Get AI Response
    bot_response = get_gemini_response(full_prompt)

    # Update chat history
    chat_sessions.setdefault(session_id, []).append({"role": "user", "content": user_query})
    chat_sessions[session_id].append({"role": "assistant", "content": bot_response})

    return jsonify({"response": bot_response})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
