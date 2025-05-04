import streamlit as st
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from dotenv import load_dotenv
import os

# Environment setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)

# Model setup
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
        backend_dev_agent.as_tool(
            tool_name="Backend_Developer",
            tool_description="You are expert in backend development"
        ),
        devops_dev_agent.as_tool(
            tool_name="DevOps_Expert",
            tool_description="You are a DevOps expert"
        )
    ],
    handoff_description="Answer the user agentic ai, devops and backend development queries."
)

agent = Agent(
    name="Agent",
    instructions="You are the orchestrator. Route user queries to the most suitable agent",
    model=model,
    handoffs=[agentic_ai_agent, app_dev_agent, web_dev_agent]
)

# --- Streamlit UI Design ---
st.set_page_config(page_title="Multi-Agent Dev Assistant", page_icon="🤖", layout="centered")
st.markdown("## 🤖 Multi-Agent AI Assistant")
st.markdown("Welcome to your **intelligent developer assistant**. Ask anything about web, app, backend, or DevOps!")

# Input UI
st.markdown("### 📝 Ask your question below:")
with st.container():
    user_input = st.text_input("Type your query...", placeholder="")

# Example prompts
with st.expander("💡 Example Prompts"):
    st.markdown("""
- How do I create a REST API in Node.js?
- What is JWT authentication and how does it work?
- How to deploy a web app using Docker and Kubernetes?
- How to create a login screen in Flutter?
- How to create a responsive navbar using Tailwind CSS?
""")

# Button + Output
if st.button("🚀 Run Agent"):
    if user_input.strip():
        with st.spinner("🧠 Thinking..."):
            try:
                response = Runner.run_sync(
                    starting_agent=agent,
                    input=user_input,
                    run_config=config
                )
                st.success("✅ Response")
                st.markdown(f"```\n{response.final_output}\n```")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a question first.")

# Footer
st.markdown("---")
st.markdown("Made by Misbah using [Streamlit](https://streamlit.io/) and [OpenAI Agents SDK](https://platform.openai.com/docs/agents)")
