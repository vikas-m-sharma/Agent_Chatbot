# if you don't use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import streamlit as st
import requests

# Step 1: Setup Streamlit UI
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents 🤖")
st.write("Create and interact with your custom AI agent!")

system_prompt = st.text_area("🧠 Define your AI Agent:", height=70, placeholder="E.g., Act as a helpful assistant...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

selected_model = st.selectbox(
    "Select Model:",
    MODEL_NAMES_GROQ if provider == "Groq" else MODEL_NAMES_OPENAI
)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("💬 Ask something:", height=150, placeholder="What's the capital of India?")

API_URL = "http://127.0.0.1:9999/chat"

if st.button("🚀 Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Try to extract the actual string response
                final_response = data.get("response", data)  # handles both dict or plain string
                st.success("✅ Agent Response:")
                st.markdown(f"**{final_response}**")
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")
    else:
        st.warning("Please enter a question before submitting.")
