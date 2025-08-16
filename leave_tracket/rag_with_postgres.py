import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import AsyncOpenAI

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# SQLAlchemy Postgres setup
# ---------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
db = create_engine(DATABASE_URL, echo=True)  # echo=True → SQL logs
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db)

# ---------------------------
# ChromaDB setup (for policies)
# ---------------------------
openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_policies")
collection = chroma_client.get_or_create_collection(
    name="policies",
    embedding_function=openai_ef
)

# Insert example policy if empty
if collection.count() == 0:
    collection.upsert(
        documents=["Company policy: Employees are entitled to 20 days annual leave per year."],
        ids=["policy1"],
        metadatas=[{"type": "policy"}]
    )

# ---------------------------
# Semantic Kernel Setup
# ---------------------------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Create execution settings with function calling enabled
execution_settings = OpenAIChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(),
    max_tokens=1000,
    temperature=0.1  # Lower temperature for more consistent routing
)

chat_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=openai_client,
)

# ---------------------------
# HR Plugin Definition
# ---------------------------
class HRPlugin:
    @kernel_function(
        description="Gets the leave balance for a specific employee from the database. Use this when users ask about remaining leave days, vacation balance, or time off for a specific person."
    )
    def get_leave_balance(
        self, employee_name: Annotated[str, "The name of the employee to check leave balance for"]
    ) -> Annotated[str, "Returns the leave balance information for the employee"]:
        """Fetch leave balance from Postgres."""
        try:
            with db.connect() as conn:
                result = conn.execute(
                    text("SELECT balance FROM leaves WHERE employee = :emp"),
                    {"emp": employee_name}
                ).fetchone()
                if result:
                    return f"{employee_name} has {result[0]} days of annual leave remaining."
                return f"No leave record found for {employee_name}."
        except Exception as e:
            return f"Error retrieving leave balance: {str(e)}"

    @kernel_function(
        description="Searches company HR policies and procedures. Use this for questions about leave policies, company rules, entitlements, procedures, or general HR information."
    )
    def query_policy(
        self, query: Annotated[str, "The policy question or topic to search for"]
    ) -> Annotated[str, "Returns relevant policy information"]:
        """Fetch policy info from ChromaDB."""
        try:
            results = collection.query(query_texts=[query], n_results=1)
            if results["documents"]:
                return results["documents"][0][0]
            return "No relevant policy found."
        except Exception as e:
            return f"Error retrieving policy information: {str(e)}"

# ---------------------------
# Kernel Initialization
# ---------------------------
kernel = Kernel()
kernel.add_service(chat_service)
kernel.add_plugin(HRPlugin(), plugin_name="HRPlugin")

# ---------------------------
# Enhanced RAG Query Function
# ---------------------------
async def rag_query_semantic(user_query: str) -> str:
    """Use Semantic Kernel to intelligently route queries and generate responses."""
    try:
        chat_history = ChatHistory()
        
        # System message to set context and behavior
        chat_history.add_system_message(
            "You are an HR assistant that helps employees with leave-related questions and HR policies. "
            "ALWAYS use the available functions to get accurate, up-to-date information. "
            "For leave balance questions, use get_leave_balance with the employee name. "
            "For policy questions, use query_policy. "
            "Extract employee names from questions when needed. "
            "Provide helpful, accurate responses based on the function results. "
            "If the information is not available, clearly state that."
        )
        
        chat_history.add_user_message(user_query)

        # Get response with function calling
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )
        
        return response[0].content if response else "Sorry, I couldn't process your request."
        
    except Exception as e:
        return f"Error processing request: {str(e)}"

# ---------------------------
# Synchronous wrapper for backward compatibility
# ---------------------------
def rag_query(user_query: str, employee_name: str = None) -> str:
    """Synchronous wrapper for the semantic RAG query."""
    # If employee_name is provided, include it in the query for better context
    if employee_name and employee_name.lower() not in user_query.lower():
        enhanced_query = f"{user_query} for employee {employee_name}"
    else:
        enhanced_query = user_query
    
    return asyncio.run(rag_query_semantic(enhanced_query))

# ---------------------------
# Legacy functions (kept for compatibility)
# ---------------------------
def get_leave_balance(employee_name: str) -> str:
    """Legacy function - kept for backward compatibility."""
    hr_plugin = HRPlugin()
    return hr_plugin.get_leave_balance(employee_name)

def query_policy(query: str) -> str:
    """Legacy function - kept for backward compatibility."""
    hr_plugin = HRPlugin()
    return hr_plugin.query_policy(query)

# ---------------------------
# Interactive mode
# ---------------------------
async def interactive_hr_assistant():
    """Interactive HR assistant using Semantic Kernel."""
    print("Welcome to the HR Assistant!")
    print("Ask me about leave balances, HR policies, or company procedures.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("Thank you for using the HR Assistant. Have a great day!")
                break
                
            if not user_input:
                print("Please enter a question or type 'quit' to exit.")
                continue
            
            print(f"\n{'='*50}")
            print(f"Processing: {user_input}")
            print(f"{'='*50}")
            
            response = await rag_query_semantic(user_input)
            print(f"HR Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using the HR Assistant.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'quit' to exit.\n")

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    print("=== Testing Semantic Kernel RAG System ===\n")
    
    # Example 1: Leave balance query (Semantic Kernel will route to get_leave_balance)
    print("1. Testing leave balance query:")
    result1 = rag_query("How many leave days does Alice have left?")
    print(f"Result: {result1}\n")

    # Example 2: Policy query (Semantic Kernel will route to query_policy)
    print("2. Testing policy query:")
    result2 = rag_query("How many annual leave days are employees entitled to?")
    print(f"Result: {result2}\n")
    
    # Example 3: Interactive mode
    print("3. Starting interactive mode...")
    asyncio.run(interactive_hr_assistant())
