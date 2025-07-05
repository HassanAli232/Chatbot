import streamlit as st
import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load your custom road data class
from geojosn_reader import GeoRoadReader 

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load road info
reader = GeoRoadReader(data_dir="data")
road_info = reader.get_roads_metadata()
available_roads = [r["road"] for r in road_info]

# Create system message
road_list_str = "\n".join(f"- {r}" for r in sorted(set(available_roads)))
system_prompt = f"""
You are a helpful Riyadh roads assistant trained only on a limited set of roads.

Your job is to:
- Answer questions only about:
    - Roads in the dataset
    - Your identity and purpose
    - Your knowledge and limitations as a roads assistant
- If the user asks who you are, what you know, how you can help, or why you exist, explain that you are a roads assistant designed to help with available roads in Riyadh, and provide some examples.
- If the user asks about anything outside the scope (like sports, news, or weather), politely reject the question in *the user's prompt language*, using this phrase:
  *Sorry, I can only help you with available roads.*

⚠️ Rules:
- Always answer in the same language as the user's prompt.
- Always translate road names to the user's language.
- Never make up roads that are not listed.
- If your mixing user's language with English names, make sure it is typed clearly.
- If the user asks about a road not in the dataset, politely inform them that you don't know the specified road.

The available roads are:
{road_list_str}
"""



# Initialize Streamlit app
st.title("Riyadh Roads Chatbot")

# Initialize chat history with system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Display chat history
for message in st.session_state.messages[1:]:  # Skip system prompt
    st.chat_message(message["role"]).markdown(message["content"])

# Input from user
prompt = st.chat_input("Ask me about Riyadh's roads:")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content

        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error("There was an error while generating the response.")
        print(f"Error: {e}")
