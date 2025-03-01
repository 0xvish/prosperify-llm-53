import streamlit as st
import requests

# Streamlit UI Setup
st.set_page_config(page_title="Prosperify Chatbot", page_icon="💰")

st.title(":blue[Prosperify]")
st.write("Ask me anything about finance!")

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = requests.post("http://127.0.0.1:5000/chat", json={"query": user_input})
        
        if response.status_code == 200:
            bot_reply = response.json().get("response", "No response")
            st.markdown(f"💬 **You:** {user_input}")  # Styled User Input
            st.markdown(f"🤖 **Bot:** {bot_reply}", unsafe_allow_html=True)  # Styled Bot Response
        else:
            st.error("Error communicating with the backend.")
