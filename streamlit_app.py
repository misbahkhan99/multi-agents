import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled

# Load environment variables
load_dotenv()

# Model Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-1.5-flash"
set_tracing_disabled(True)

# Async client
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Tools
@function_tool
def handle_web_task(task: str) -> str:
    return f"[🌐 Web Dev] Working on your web development task: {task} (HTML, CSS, JS, React)."

@function_tool
def handle_app_task(task: str) -> str:
    return f"[📱 App Dev] Handling your app development task: {task} (Flutter, React Native, cross-platform)."

@function_tool
def handle_backend_task(task: str) -> str:
    return f"[🛠 Backend Dev] Working on backend task: {task} (API, DB, auth, server logic)."

@function_tool
def handle_devops_task(task: str) -> str:
    return f"[⚙️ DevOps] Handling DevOps task: {task} (CI/CD, server deployment, containers, infra-as-code)."

# Agents
web_dev_agent = Agent(
    name="Web Development Agent",
    instructions="You are an expert front-end web developer...",
    model=model,
    tools=[handle_web_task],
    handoff_description="Handles all user queries related to websites and frontend development."
)

app_dev_agent = Agent(
    name="App Development Agent",
    instructions="You are an expert in mobile and cross-platform app development...",
    model=model,
    tools=[handle_app_task],
    handoff_description="Handles all user queries related to app development and mobile UI."
)

backend_dev_agent = Agent(
    name="Backend_Developer_Agent",
    instructions="You are an expert in backend development...",
    model=model,
    tools=[handle_backend_task],
    handoff_description="Handles backend APIs, databases, auth systems, and server-side programming."
)

devops_dev_agent = Agent(
    name="DevOps_Developer_Agent",
    instructions="You are a DevOps specialist...",
    model=model,
    tools=[handle_devops_task],
    handoff_description="Expert in CI/CD, server deployment, containerization, and cloud infrastructure."
)

agentic_ai_agent = Agent(
    name="Agentic_AI_Agent",
    instructions="Decides whether the task is backend or DevOps...",
    model=model,
    tools=[
        backend_dev_agent.as_tool("Backend_Developer", "You are expert in backend development"),
        devops_dev_agent.as_tool("DevOps_Expert", "You are a DevOps expert")
    ],
    handoff_description="Answer the user agentic ai, devops and backend development queries."
)

# Orchestrator Agent
main_agent = Agent(
    name="Agent",
    instructions="You are the orchestrator. Route user queries to the most suitable agent",
    model=model,
    handoffs=[agentic_ai_agent, app_dev_agent, web_dev_agent]
)

# Runner Function
async def run_agent_async(user_input):
    response = await Runner.run(
        starting_agent=main_agent,
        input=user_input,
        run_config=config
    )
    return response.final_output

def run_async_wrapper(user_input):
    return asyncio.run(run_agent_async(user_input))

# ----------------- Streamlit UI --------------------

st.set_page_config(page_title="🧠 AI Developer Assistant", page_icon="🤖", layout="centered")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/robot-2.png", width=80)
    st.title("AI Dev Assistant")
    st.markdown("""
    🚀 Ask me anything related to:
    - Web Dev
    - App Dev
    - Backend
    - DevOps
    - Agentic AI

    Created by **Misbah**
    """)
    st.markdown("---")
    st.caption("Powered by Gemini 1.5 Flash ✨")

# Main UI
st.markdown("""
    <div style="text-align:center">
        <h2 style="color:#4A90E2">Welcome to the AI Developer Assistant</h2>
        <p style="font-size:16px;">A multi-agent system that routes your questions to the best expert assistant.</p>
    </div>
""", unsafe_allow_html=True)

with st.container():
    user_input = st.text_input("🔍 What would you like help with today?", placeholder="E.g. what is web development")

    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        if st.button("💬 Submit Query", use_container_width=True):
            if user_input.strip():
                with st.spinner("🤖 Thinking..."):
                    result = run_async_wrapper(user_input)
                    st.markdown("### ✅ Response")
                    st.success(result)
            else:
                st.warning("Please enter a valid query.")

# Footer
st.markdown("""<hr style="margin-top:40px">""", unsafe_allow_html=True)
st.caption("© 2025 AI Agentic Systems — All rights reserved.")
