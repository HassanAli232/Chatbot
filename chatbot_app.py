import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from RAG_helper_functions import get_roads_context
from road_vector_index import RoadVectorDB
from geojosn_reader import GeoRoadReader

# === Debug display helper ===
def printDebug(title, content):
    with st.expander("🔍 Debug: " + title):
        st.code(content, language="markdown")


# === Load once at startup ===
@st.cache_resource
def startup():
    
    load_dotenv()

    # Initialize OpenAI client and GeoRoadReader
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    reader = GeoRoadReader(data_dir="data")

    # Load road metadata
    road_info = reader.get_roads_metadata()
    available_roads = list(set(r["road"] for r in road_info))

    # Build FAISS vector DB
    vector_db = RoadVectorDB()
    vector_db.build_index(available_roads, client)

    return client, reader, available_roads, vector_db


# === Static system prompt ===
def build_base_prompt(available_roads):
    road_list_str = "\n".join(f"- {r}" for r in sorted(set(available_roads)))
    return f"""
You are a helpful Riyadh roads assistant trained only on a limited set of roads.

Your job is to answer questions related only to:
    - Your identity and purpose
    - Roads in the dataset or related information
    - Road names, distances, speed limits, and travel times, etc.
    - Your knowledge and limitations as a roads assistant
Guidelines:
    - If the user asks who you are, what you know, how you can help, or why you exist, explain that you are a roads assistant designed to help with available roads in Riyadh, and provide some examples.
    - If the user asks about anything outside your job (like sports, news, or weather), politely reject the question, using this phrase:
  *Sorry, I can only help you with available roads.*
    - If the user does **not specify a road**, you may choose any known road from the list to use as an example or to illustrate a general point.
    - You may also use general trends across all roads if the user asks for statistics or examples about "roads" in general.

Rules:
    - Never make up roads that are not listed.
    - Make sure the response is typed clearly.
    - If the user asks about a road not in the dataset, find the most similar road name in the dataset.

Notes:
    - Answer all questions related to roads as long as they are in the road context.
    - You are not required to say who you are unless the user asks.
    - Each road has different versions based on the year, so always use the latest version of that road unless the user specifys.

The available roads are:
{road_list_str}

All information about the roads should be based on the following road contexts:
""".strip()


def check_year(prompt):
    """
    Check if the user prompt contains a year.
    Returns True if a year is mentioned, otherwise False.
    """
    import re

    prompt = prompt.strip().lower()
    # Check for years in the format YYYY or YY
    return bool(re.search(r'\b(20\d{2}|[0-9]{2})\b', prompt)) or any(word in prompt for word in ["year", "versions", "version", "years"])


# === Main Chat Handling ===
def handle_user_prompt(prompt, client, reader, vector_db, base_prompt):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Find the best road match using vector DB
    matched = vector_db.search(prompt, client)
    
    # Debugging output
    printDebug("Matched Roads", matched or "No roads matched.")

    # Check if the user specified a year in their prompt
    require_year = check_year(prompt)
    printDebug("Require Year", str(require_year))

    # For roads' context
    road_context = "\n\nNo matching road found for the user's prompt."
    if matched:
        road_context = get_roads_context(matched, reader=reader, versions=require_year)
    printDebug("Road Context", road_context)

    # Combine the base prompt with the road context
    final_system_prompt = base_prompt + "\n\n" + road_context

    # Prepare the messages for the OpenAI API
    messages = [{"role": "system", "content": final_system_prompt}] + st.session_state.messages
    printDebug("final_system_prompt", final_system_prompt)
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


# === Entry Point ===
if __name__ == "__main__":
    st.set_page_config(page_title="Riyadh Roads Chatbot", page_icon="🚗")

    client, reader, available_roads, vector_db = startup()
    base_system_prompt = build_base_prompt(available_roads)

    st.title("Riyadh Roads Chatbot")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Lodad messages from session state
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # User input for chat
    prompt = st.chat_input("Ask me about Riyadh's roads:")
    if prompt:
        handle_user_prompt(prompt, client, reader, vector_db, base_system_prompt)
