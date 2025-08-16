import os
import asyncio
import json
from typing import Annotated
from dotenv import load_dotenv

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function


# -----------------------------
# 1. Plugin Definition
# -----------------------------
class DestinationsPlugin:
    @kernel_function(description="Provides a list of vacation destinations.")
    def get_destinations(self) -> Annotated[str, "Returns the vacation destinations."]:
        return """
        Barcelona, Spain
        Paris, France
        Berlin, Germany
        Tokyo, Japan
        New York, USA
        """

    @kernel_function(description="Provides the availability of a destination.")
    def get_availability(
        self, destination: Annotated[str, "The destination to check availability for."]
    ) -> Annotated[str, "Returns the availability of the destination."]:
        return """
        Barcelona - Unavailable
        Paris - Available
        Berlin - Available
        Tokyo - Unavailable
        New York - Available
        """

class WeatherPlugin:
    @kernel_function(description="Provides the weather for a destination.")
    def get_weather(self, destination: Annotated[str, "The destination to check weather for."]) -> Annotated[str, "Returns the weather for the destination."]:
        return """
        Barcelona - Sunny
        Paris - Cloudy
        Berlin - Rainy
        Tokyo - Rainy
        New York - Sunny
        """

class TripPlannerPlugin:
    @kernel_function(description="Creates a detailed trip plan for a destination.")
    def create_trip_plan(self, destination: Annotated[str, "The destination to create a trip plan for."]) -> Annotated[str, "Returns a detailed trip plan for the destination."]:
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


# -----------------------------
# 2. Setup Environment & AI Service
# -----------------------------
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create execution settings with function calling enabled
execution_settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(),
    max_tokens=1000,
    temperature=0.7
)

chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=openai_client,
)

# -----------------------------
# 3. Kernel Initialization & Plugin Registration
# -----------------------------
kernel = Kernel()
kernel.add_service(chat_service)
kernel.add_plugin(DestinationsPlugin(), plugin_name="DestinationsPlugin")
kernel.add_plugin(WeatherPlugin(), plugin_name="WeatherPlugin")
kernel.add_plugin(TripPlannerPlugin(), plugin_name="TripPlannerPlugin")

# -----------------------------
# 4. Interactive Function Calling Test
# -----------------------------
async def main():
    print("Welcome to the Travel Agent Assistant!")
    print("Ask me about destinations and their availability.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        chat_history = ChatHistory()
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit conditions
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("Thank you for using the Travel Agent Assistant. Have a great day!")
                    break
                    
                if not user_input:
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                
                print(f"\n{'='*50}")
                print(f"User: {user_input}")
                print(f"{'='*50}")

                # Create a chat history and add the user message
                
                chat_history.add_system_message("You are a proactive travel agent. ALWAYS use the available functions to get current destination information, availability, and weather data. Never rely on general knowledge - always call the functions to get accurate, up-to-date information. When a user asks for trip planning with specific criteria, call all relevant functions, analyze the results, automatically select the best option that meets their requirements, and provide a complete travel plan. Be decisive and take immediate action without asking for confirmation.")
                chat_history.add_user_message(user_input)

                # Invoke the chat completion with function calling enabled
                response = await chat_service.get_chat_message_contents(
                    chat_history=chat_history,
                    settings=execution_settings,
                    kernel=kernel
                )
                
                # Check if there were any function calls in the response
                for message in response:
                    if hasattr(message, 'items') and message.items:
                        print("\n--- Function Calls Made ---")
                        for item in message.items:
                            if hasattr(item, 'function_name') and hasattr(item, 'arguments'):
                                print(f"Called: {item.function_name}({item.arguments})")
                            elif hasattr(item, 'result'):
                                print(f"Result: {item.result}")
                        print("--- End Function Calls ---\n")
                
                print(f"Assistant: {response[0].content}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Thanks for using the Travel Agent Assistant.")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or type 'quit' to exit.\n")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye! Thanks for using the Travel Agent Assistant.")


# Let's create a synchronous wrapper for testing
def run_main():
    """Synchronous wrapper to run the async main function"""
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error running main: {e}")

# Test the code
if __name__ == "__main__":
    run_main()
else:
    # For Jupyter notebook, we can call this directly
    print("Ready to run. Call run_main() to execute the agent.") 