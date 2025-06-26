# Problem:

You want an AI agent to:
- Take a research question
- Search the web
- Extract and summarize the top 3 answers

Save the summary to a file

This mimics an agentic AI behavior: it has a goal, autonomy, and a tool-use loop to reason and act.

## What Makes This Agentic?

Goal-Oriented: It tries to find the best answer to your query.

Autonomous: You only give a goal, not step-by-step commands.

Tool Use: It invokes search tools without hardcoding responses.

Planning/Reasoning: LangChain agents use reasoning chains to decide which tool to call and when.