# if you don't use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step 1: Setup API keys for Groq, OpenAI, and Tavily
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

# ✅ Import LangChain message types
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Step 2: Setup default LLMs (optional)
openai_llm = ChatOpenAI(model="gpt-4o-mini")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step 3: Core agent function
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    # Select model
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        return "Invalid provider"

    # Optional search tool
    tools = [TavilySearch(max_results=2)] if allow_search else []

    # Create the agent (no system_prompt argument anymore)
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # Inject the system prompt + user query into the message list
    messages = [SystemMessage(content=system_prompt)]
    messages += [HumanMessage(content=q) for q in query]

    # Invoke agent
    state = {"messages": messages}
    response = agent.invoke(state)

    # Extract AI message
    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

    return ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."
