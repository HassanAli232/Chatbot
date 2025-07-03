import streamlit as st
import openai
import os
from openai import OpenAI
# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Create OpenAI client (v1+ style)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("Chatbot App")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask me something:")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Use new v1 API format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content

        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        raise "Error"

    except Exception as e:
        st.error(f"There is an error")
        print(f"Error: {e}")  
