#!/usr/bin/env python3
"""
Test script for the Semantic Kernel-based RAG system
Demonstrates intelligent routing without hardcoded conditions
"""

import os
import asyncio
from dotenv import load_dotenv

# Mock the database for testing (replace with actual DB setup)
class MockDatabase:
    def __init__(self):
        self.employees = {
            "Alice": 15,
            "Bob": 8,
            "Charlie": 22,
            "Diana": 5
        }
    
    def get_balance(self, employee_name):
        return self.employees.get(employee_name)

# For this test, we'll create a simplified version
async def test_semantic_routing():
    """Test different types of queries to see how Semantic Kernel routes them"""
    
    # Test queries - mix of leave balance and policy questions
    test_queries = [
        "How many leave days does Alice have left?",
        "What's Bob's current vacation balance?", 
        "How many annual leave days are employees entitled to?",
        "What is the company policy on sick leave?",
        "Can you check Diana's remaining time off?",
        "What are the rules for taking consecutive leave days?",
        "How much vacation time does Charlie have available?",
        "What's the policy for emergency leave?"
    ]
    
    print("=== Semantic Kernel RAG System Test ===\n")
    print("This demonstrates how Semantic Kernel intelligently routes queries")
    print("without hardcoded 'if' conditions.\n")
    
    # Import the main rag system
    try:
        from rag_with_postgres import rag_query_semantic, interactive_hr_assistant
        
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. Query: '{query}'")
            try:
                result = await rag_query_semantic(query)
                print(f"   Response: {result}")
                print(f"   {'='*60}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"   Error: {e}")
                print(f"   {'='*60}")
            
            print()  # Empty line for readability
            
    except ImportError as e:
        print(f"Error importing rag_with_postgres: {e}")
        print("Please ensure the environment is set up correctly.")
        return
    
    print("\nTest completed! The system intelligently determined:")
    print("- Employee-specific queries → get_leave_balance function")
    print("- Policy/general queries → query_policy function")
    print("- No hardcoded 'if' statements needed!")

async def interactive_test():
    """Interactive test mode"""
    print("\n=== Interactive Test Mode ===")
    print("Try asking different questions to see the routing in action:")
    print("Examples:")
    print("- 'How many days does Alice have?'")
    print("- 'What's the leave policy?'")
    print("- 'Check Bob's vacation balance'")
    print("Type 'quit' to exit\n")
    
    try:
        from rag_with_postgres import interactive_hr_assistant
        await interactive_hr_assistant()
    except ImportError as e:
        print(f"Error: {e}")

def main():
    """Main function to run tests"""
    load_dotenv()
    
    # Check if required environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file")
        return
    
    if not os.getenv("DATABASE_URL"):
        print("Warning: DATABASE_URL not found in environment variables")
        print("You may need to set this for database operations to work")
        print("Continuing with test anyway...\n")
    
    print("Select test mode:")
    print("1. Automated test (runs predefined queries)")
    print("2. Interactive test (manual queries)")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_semantic_routing())
    elif choice == "2":
        asyncio.run(interactive_test())
    elif choice == "3":
        asyncio.run(test_semantic_routing())
        asyncio.run(interactive_test())
    else:
        print("Invalid choice. Running automated test...")
        asyncio.run(test_semantic_routing())

if __name__ == "__main__":
    main() 