# pip install langchain langgraph openai python-dotenv
import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 1) Tools (match your plugin methods)
# -----------------------------
@tool
def get_destinations() -> str:
    """Provides a list of vacation destinations."""
    return """
Barcelona, Spain
Paris, France
Berlin, Germany
Tokyo, Japan
New York, USA
""".strip()

@tool
def get_availability(destination: str) -> str:
    """Provides the availability of a destination."""
    return """
Barcelona - Unavailable
Paris - Available
Berlin - Available
Tokyo - Unavailable
New York - Available
""".strip()

@tool
def get_weather(destination: str) -> str:
    """Provides the weather for a destination."""
    return """
Barcelona - Sunny
Paris - Cloudy
Berlin - Rainy
Tokyo - Rainy
New York - Sunny
""".strip()

@tool
def create_trip_plan(destination: str) -> str:
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

Day 3: Culture & Parks
- Central Park & Metropolitan Museum
- Shopping on 5th Avenue
- Evening show on Broadway

🏨 ACCOMMODATION: Midtown Manhattan hotels
🍽️ FOOD: Try NYC pizza, bagels, and diverse cuisines
🚇 TRANSPORT: Metro Card for subway system
💰 BUDGET: $150-300/day depending on preferences
""",
        "Barcelona": """
🏛️ BARCELONA TRIP PLAN 🏛️

📅 ITINERARY:
Day 1: Gothic Quarter & Las Ramblas
Day 2: Sagrada Familia & Park Güell
Day 3: Beach & Barceloneta

🏨 ACCOMMODATION: Gothic Quarter or Eixample
🍽️ FOOD: Tapas, paella, sangria
🚇 TRANSPORT: Metro and walking
""",
        "Paris": """
🗼 PARIS TRIP PLAN 🗼

📅 ITINERARY:
Day 1: Eiffel Tower & Seine River
Day 2: Louvre & Champs-Élysées
Day 3: Montmartre & Sacré-Cœur

🏨 ACCOMMODATION: Near Metro stations
🍽️ FOOD: French cuisine, cafes, pastries
"""
    }
    return plans.get(destination, f"Trip plan for {destination} - Contact local tourism office for detailed itinerary.")

TOOLS = [get_destinations, get_availability, get_weather, create_trip_plan]

# -----------------------------
# 2) LangGraph state + nodes
# -----------------------------
class State(TypedDict):
    messages: List

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)

# System instruction like your SK system message
SYSTEM = (
    "You are a proactive travel agent. ALWAYS use the available functions to get "
    "current destination information, availability, and weather data. Never rely on general "
    "knowledge—always call the functions. When a user asks for trip planning with specific "
    "criteria, call all relevant functions, analyze results, select the best option, and "
    "provide a complete plan. Be decisive and take immediate action without asking for confirmation."
)

# The "agent" node: call the model with tools bound
def call_model(state: State):
    # Bind tools for tool-calling
    model = llm.bind_tools(TOOLS)
    # Prepend system message once; model will see all prior messages
    msgs = [("system", SYSTEM)] + state["messages"]
    response = model.invoke(msgs)
    return {"messages": [response]}

# Router: if the last AI message has tool calls, go to ToolNode; else end
def route(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END

# Build the graph
graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge("tools", "agent")       # after tools run, go back to agent
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route)

memory = MemorySaver()  # optional (keeps per-session state)
app = graph.compile(checkpointer=memory)

# -----------------------------
# 3) Simple REPL
# -----------------------------
def run():
    print("Welcome to the Travel Agent Assistant (LangGraph)!")
    config = {"configurable": {"thread_id": "travel_session"}}
    
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("Assistant: Have a great day!")
            break
        if not user:
            print("Please enter a question or type 'quit' to exit.")
            continue

        # Create input with user message
        input_state = {"messages": [HumanMessage(content=user)]}
        
        # Invoke the graph - it will handle the full conversation flow
        result = app.invoke(input_state, config=config)

        # Print latest assistant reply
        last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        print(f"Assistant: {last_ai.content if last_ai else '(no reply)'}\n")

if __name__ == "__main__":
    run()
