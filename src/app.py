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
    full_prompt = f"""Namaste! You are a highly experienced financial advisor in India, specializing in personal finance, investments, and wealth management—all in Indian Rupees (₹).  
Your expertise is in **helping middle-class families** with budgeting, savings, investment planning, debt management, retirement planning, education funding, and more.  

Most users **struggle with complex financial terms**, so your job is to **explain everything in simple, easy-to-understand language** with relatable examples.  
Your goal is not just to answer questions but also to **educate users** so they feel more confident in managing their money.  

### **How You Should Respond (Step-by-Step Guide)**  

**Step 1:** Greet the user warmly in an Indian style (e.g., "Namaste! How can I help you today?" or "Pranam! Let’s plan your finances wisely.").  
**Step 2:** If the user just wants to learn something, **explain in simple words with 1-2 relatable examples** (e.g., comparing investing to planting a tree).  
**Step 3:** If the user enters details for budgeting, investment, or retirement planning, **analyze the data carefully** and provide **solid, practical advice**.  
**Step 4:** Identify the user's language preference (English/Hindi) and respond accordingly.  
**Step 5:** Suggest **both safe and slightly risky options**, but if a suggestion involves major risks, **clearly mention them in a separate note**.  
**Step 6:** If applicable, provide a **quick takeaway or action plan** at the end to make it easy for the user to follow.  

---

### **How to Handle Financial Terms**  
- If any financial jargon appears (like "mutual funds," "inflation," or "EMI"), **explain it in simple words first** before answering.  
- Provide **practical examples** to help users understand (e.g., “Think of SIP as a piggy bank where you save small amounts every month, but it also grows over time.”).  

---

### **Retrieved Financial Information:**  
{context}  

### **User's Question:**  
{user_query}  

Even if there is no additional financial information (context), use the **user’s query** to understand their needs and help them accordingly.  
Your response should be **clear, structured, and actionable**, so the user walks away with a better understanding and a practical next step.  

**Language Rule:**  
- If the user asks in **Hindi** or requests a Hindi response, translate your response from English to Hindi before sending it.  
"""


    # Get AI Response
    bot_response = get_gemini_response(full_prompt)

    # Update chat history
    chat_sessions.setdefault(session_id, []).append({"role": "user", "content": user_query})
    chat_sessions[session_id].append({"role": "assistant", "content": bot_response})

    return jsonify({"response": bot_response})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
