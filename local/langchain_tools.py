# pip install langchain langchain-openai python-dotenv
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Tools ----------
@tool
def get_destinations() -> Annotated[str, "Returns the vacation destinations."]:
    """Provides a list of vacation destinations."""
    return """Barcelona, Spain
Paris, France
Berlin, Germany
Tokyo, Japan
New York, USA"""

@tool
def get_availability(destination: Annotated[str, "The destination to check availability for."]) -> Annotated[str, "Returns the availability of the destination."]:
    """Provides the availability of a destination."""
    return """Barcelona - Unavailable
Paris - Available
Berlin - Available
Tokyo - Unavailable
New York - Available"""

@tool
def get_weather(destination: Annotated[str, "The destination to check weather for."]) -> Annotated[str, "Returns the weather for the destination."]:
    """Provides the weather for a destination."""
    return """Barcelona - Sunny
Paris - Cloudy
Berlin - Rainy
Tokyo - Rainy
New York - Sunny"""

@tool
def create_trip_plan(destination: Annotated[str, "The destination to create a trip plan for."]) -> Annotated[str, "Returns a detailed trip plan for the destination."]:
    """Creates a detailed trip plan for a destination."""
    plans = {
            "New York": """
            🗽 NEW YORK TRIP PLAN 🗽
            
            📅 ITINERARY:
            Day 1: Arrive & Manhattan Tour
            - Check into hotel in Midtown
            - Visit Times Square & Broadway area
            - Dinner in Little Italy
            
            Day 2: Iconic Landmarks
            - Statue of Liberty & Ellis Island (morning)
            - 9/11 Memorial & One World Observatory
            - Walk across Brooklyn Bridge (sunset)
            - Dinner in Chinatown
            - Broadway show (optional)
            """
        }
    return plans.get(destination, "No trip plan found for this destination.")

tools = [get_destinations, get_availability, get_weather, create_trip_plan]

SYSTEM = (
    "You are a proactive travel agent. ALWAYS use the available functions to get "
    "current destination information, availability, and weather data. Never rely on general "
    "knowledge—always call the functions. When a user asks for trip planning with specific "
    "criteria, call all relevant functions, analyze results, select the best option, and "
    "provide a complete plan. Be decisive."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run():
    print("Welcome to the Travel Agent Assistant (LangChain)!")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("Assistant: Have a great day!")
            break
        if not user:
            print("Please enter a question or type 'quit' to exit.")
            continue

        out = executor.invoke({"input": user})
        print("Assistant:", out["output"], "\n")

if __name__ == "__main__":
    run()
