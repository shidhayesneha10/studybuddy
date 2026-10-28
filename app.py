
import json
import os
from datetime import datetime

import streamlit as st
from openai import OpenAI

# ---------- Setup ----------
st.set_page_config(page_title="StudyBuddy AI Agent", page_icon="🎓")
def get_setting(name, default=None):
    """Read from environment variables first, then Streamlit secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets[name]
    except Exception:
        return default


# Works with OpenAI, Groq or Gemini (all OpenAI-compatible). Defaults to OpenAI.
api_key = get_setting("LLM_API_KEY") or get_setting("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Set LLM_API_KEY (or OPENAI_API_KEY) as an environment variable or in Streamlit secrets.")
    st.stop()
BASE_URL = get_setting("LLM_BASE_URL")          # None = OpenAI
MODEL = "gemini-3.8-flash"
client = OpenAI(api_key=api_key, base_url=BASE_URL)

# ---------- Prompt engineering: one system prompt per mode ----------
MODES = {
    "Explain a concept": "Explain the topic in simple language with one real-life example and a short summary at the end.",
    "Quiz me": "Create 5 multiple-choice questions on the topic. Give answers only after the student replies.",
    "Make revision notes": "Produce concise bullet-point revision notes with key terms in bold.",
    "Study planner": "Create a day-wise study plan based on the subject, days left and hours per day given by the student.",
}
BASE_PROMPT = (
    "You are StudyBuddy, a friendly AI study assistant for engineering students. "
    "Be accurate, concise and encouraging. If unsure, say so. Use tools when useful."
)

# ---------- Tools the agent can call ----------
def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "Invalid expression"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def days_until(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return f"{(d - datetime.now().date()).days} days remaining"
    except ValueError:
        return "Use format YYYY-MM-DD"


TOOL_FUNCS = {"calculator": calculator, "days_until": days_until}
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate a math expression",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "days_until", "description": "Days left until a date (YYYY-MM-DD), e.g. an exam date",
        "parameters": {"type": "object",
                       "properties": {"date_str": {"type": "string"}},
                       "required": ["date_str"]}}},
]

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.selectbox("Choose mode", list(MODES))
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.5, 0.1)
if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.pop("history", None)
    st.rerun()

# ---------- Memory: conversation stored in session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎓 StudyBuddy – AI Study Assistant")
st.caption(f"Mode: {mode}")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Agent loop ----------
def run_agent(user_text: str) -> str:
    system = f"{BASE_PROMPT} Current mode: {mode}. {MODES[mode]}"
    messages = [{"role": "system", "content": system}] + st.session_state.history[-20:]
    for _ in range(3):  # allow up to 3 tool-call rounds
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=temperature
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = TOOL_FUNCS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Sorry, I could not complete that request."


if prompt := st.chat_input("Ask a question, e.g. 'Explain overfitting'"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(prompt)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})