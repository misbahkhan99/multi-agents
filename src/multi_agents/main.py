from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from dotenv import load_dotenv
load_dotenv()
import os


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-1.5-flash"
set_tracing_disabled(True)

# model configuration
client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-1.5-flash",
    openai_client = client
)

config = RunConfig(
    model = model,
    model_provider = client, 
    tracing_disabled= True
)

# tools
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



#agents
web_dev_agent = Agent(
    name="Web Development Agent",
    instructions="You are an expert front-end web developer. Your job is to handle all web development-related tasks. Use HTML, CSS, JavaScript, React, and Tailwind CSS where needed. ONLY respond to web development queries.",
    model = model,
    tools=[handle_web_task],
    handoff_description="Handles all user queries related to websites and frontend development."
)

app_dev_agent = Agent(
    name="App Development Agent",
    instructions="You are an expert in mobile and cross-platform app development. You ONLY handle tasks related to mobile app design, development, and debugging using Flutter or React Native.",
    model = model,
    tools=[handle_app_task],
    handoff_description="Handles all user queries related to app development and mobile UI."
)


backend_dev_agent = Agent(
    name="Backend_Developer_Agent",
    instructions="You are an expert in backend development. Your job is to handle all tasks related to server-side logic, API development, database queries, and authentication systems. ONLY respond to backend-related queries.",
    model = model,
    tools=[handle_backend_task],
    handoff_description="Handles backend APIs, databases, auth systems, and server-side programming."
)

devops_dev_agent = Agent(
    name="DevOps_Developer_Agent",
    instructions="You are a DevOps specialist. Your responsibility is to manage CI/CD pipelines, automate deployments, configure servers, and handle infrastructure. ONLY respond to DevOps-related queries.",
    model = model,
    tools=[handle_devops_task],
    handoff_description="Expert in CI/CD, server deployment, containerization, and cloud infrastructure."
)

agentic_ai_agent = Agent(
    name="Agentic_AI_Agent",
    instructions="Decides whether the task is backend or DevOps and uses the correct developer agent as a tool. and also answer the agentic ai user queries",
    model = model,
    tools=[
        backend_dev_agent.as_tool(
            tool_name= "Backend_Developer",
            tool_description = "You are expert in backend development"
        ),
        devops_dev_agent.as_tool(
            tool_name = "DevOps_Expert",
            tool_description = "You are a DevOps expert"
        )
        ],
    handoff_description="Answer the user agentic ai, devops and backend deveelopment queries."
)

# triage agent
agent = Agent(
    name = "Agent",
    instructions = "You are the orchestrator. Route user queries to the most suitable agent",
    model = model,
    handoffs=[agentic_ai_agent, app_dev_agent, web_dev_agent]
)



# result
response = Runner.run_sync(
    starting_agent = agent,
    input = "what is ai agents", 
    run_config= config
)

result = response.final_output
print(result)

