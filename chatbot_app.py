import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from RAG_helper_functions import get_best_road_match, get_road_context
from geojosn_reader import GeoRoadReader

def printDebug(title, content):
    # Show the road context (debugging or developer view)
    with st.expander("🔍 Debug: " + title):
        st.code(content, language="markdown")

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load road info
reader = GeoRoadReader(data_dir="data")
road_info = reader.get_roads_metadata()
available_roads = [r["road"] for r in road_info]

# Create road list string for prompt
road_list_str = "\n".join(f"- {r}" for r in sorted(set(available_roads)))


# === Static system prompt ===
base_system_prompt = f"""
You are a helpful Riyadh roads assistant trained only on a limited set of roads.

Your job is to:
- Answer questions related only to:
    - Roads in the dataset or related information
    - Road names, distances, speed limits, and travel times, etc.
    - Your identity and purpose
    - Your knowledge and limitations as a roads assistant
- If the user asks who you are, what you know, how you can help, or why you exist, explain that you are a roads assistant designed to help with available roads in Riyadh, and provide some examples.
- If the user asks about anything outside the scope (like sports, news, or weather), politely reject the question in *the user's prompt language*, using this phrase:
  *Sorry, I can only help you with available roads.*

⚠️ Rules:
- Never make up roads that are not listed.
- Make sure the respnse is typed clearly.
- If the user asks about a road not in the dataset, politely inform them that you don't know the specified road.

Notes:
- Roads names do not need to be exactly as they appear in the dataset, but should be close enough to match.
- You are not required to say who you are unless the user asks.

The available roads are:
{road_list_str}
"""


# === Streamlit UI ===
st.title("Riyadh Roads Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle user input
prompt = st.chat_input("Ask me about Riyadh's roads:")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === RAG: try to retrieve relevant road ===
    matched = get_best_road_match(prompt, available_roads=available_roads)
    road_context = ""

    if matched:
        context_data = get_road_context(matched[0], reader=reader)
        if context_data:
            road_context = f"\n\nThe user might be referring to the road \"{matched[0]}\". Here is the known data:\n\n{context_data}"
        else:
            road_context = f"\n\nThe road \"{matched[0]}\" was matched but no data was found."
    
    # Show the road context (debugging or developer view)
    with st.expander("🔍 Debug: Road Context"):
        st.code(road_context or "No road context found.", language="markdown")
        # st.code(context_data, language="markdown")

    # === Compose final system message ===
    final_system_prompt = base_system_prompt + road_context

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error("There was an error generating the response.")
        print(f"Error: {e}")
