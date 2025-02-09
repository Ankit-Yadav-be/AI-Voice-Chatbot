import requests
import streamlit as st
import speech_recognition as sr
from langgraph.graph import StateGraph
from typing import TypedDict


gemini_api_key = "AIzaSyANKZXIvvJGTe1ZRK4CAZ_ilfsRkjOxnv4"

class ChatState(TypedDict):
    message: str

def get_gemini_response(state: ChatState) -> ChatState:
    prompt = state["message"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}  

    response = requests.post(url, json=data, headers=headers)
    
    try:
        response_json = response.json()
        
        if "candidates" in response_json:
            return {"message": response_json["candidates"][0]["content"]["parts"][0]["text"]}
        elif "error" in response_json:
            return {"message": f"API Error: {response_json['error']['message']}"}
        else:
            return {"message": "Unexpected API Response!"}
    except Exception as e:
        return {"message": f"Exception Occurred: {str(e)}"}

graph = StateGraph(ChatState)  
graph.add_node("user_input", get_gemini_response)
graph.set_entry_point("user_input")
workflow = graph.compile()


st.set_page_config(page_title="  AI voice chatbot", page_icon="🎙️", layout="centered")

st.title("🎙️Ankit Yadav Personal AI voice chatbot")
st.write("Speak to ask your questions! (Powered by Google Gemini llm model & LangGraph)")

if "messages" not in st.session_state:
    st.session_state.messages = []


for chat in st.session_state.messages:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio. Please try again."
        except sr.RequestError:
            return "Could not request results. Check your internet connection."

if st.button("🎙️ Speak Now"):
    user_input = recognize_speech()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = workflow.invoke({"message": user_input})
    bot_reply = response["message"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
