# install command
# pip install langchain openai duckduckgo-search

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Define tools the agent can use
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for answering general questions by searching the web"
    )
]

# Load the LLM
llm = OpenAI(temperature=0)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Ask the agent a question
question = "What are the latest breakthroughs in Alzheimer's disease research in 2024?"

# Agent will plan and use tools to answer this
response = agent.run(question)

# Save the output to a file
with open("research_summary.txt", "w") as f:
    f.write(response)

print("Summary saved to research_summary.txt")

