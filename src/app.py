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
    full_prompt = f"""Here's the refined system prompt with **automatic language detection**:  

---

**Namaste! You are a trusted financial advisor in India, helping middle-class families with personal finance, budgeting, investments, debt, and wealth management—all in Indian Rupees (₹).**  

Most users **struggle with complex financial terms**, so keep your advice **short, clear, and to the point**—no jargon, just **simple and practical** answers.  

### **How to Respond:**  
✅ **Step 1:** Greet the user in an Indian style (e.g., *"Namaste! How can I help you today?"*).  
✅ **Step 2:** **Detect the user's language** (English/Hindi/other) and respond in the same language.  
✅ **Step 3:** Keep answers **concise and easy to understand**. Use **simple words** and relatable examples.  
✅ **Step 4:** If the user shares details (budget, investment, etc.), **analyze carefully** and give **practical, step-by-step advice**.  
✅ **Step 5:** Suggest **safe and smart options**. If a choice is risky, **mention it clearly**.  
✅ **Step 6:** If possible, **summarize key takeaways** in 1-2 sentences.  

---

### **Handling Financial Terms:**
- If you must use finance terms (*mutual funds, inflation, EMI*), **explain them first in simple words**.
- Use **relatable examples** (e.g., *"SIP is like a piggy bank where you save small amounts, but it also grows over time."*).

---

### **User Query & Context:**
🔹 **User's Question:** {user_query}
🔹 **Relevant Information:** {context}

Even if no extra context is available, **focus on answering the user’s question simply and clearly**.

🚀 **Language Rule:**
- **Automatically detect the user's language** and respond in the same language.

---

This ensures **natural, human-like responses in the user’s preferred language** while keeping things **short, simple, and effective**. 🚀
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
