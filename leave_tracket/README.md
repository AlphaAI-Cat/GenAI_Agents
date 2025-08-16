# Semantic Kernel RAG System

This project demonstrates how to replace hardcoded `if` conditions with intelligent semantic routing using Microsoft's Semantic Kernel framework.

## What Changed?

### Before (Hardcoded Routing)
```python
# Old approach with hardcoded conditions
if "balance" in user_query.lower() and employee_name:
    context = get_leave_balance(employee_name)
else:
    context = query_policy(user_query)
```

### After (Semantic Kernel Routing)
```python
# New approach - AI automatically chooses the right function
@kernel_function(description="Gets leave balance for an employee...")
def get_leave_balance(self, employee_name: str) -> str:
    # Implementation

@kernel_function(description="Searches company HR policies...")  
def query_policy(self, query: str) -> str:
    # Implementation
```

## Key Benefits

1. **Intelligent Routing**: AI understands user intent and calls the appropriate function
2. **No Hardcoded Rules**: No need to guess keywords or write complex if-statements
3. **Natural Language**: Users can ask questions naturally without specific keywords
4. **Extensible**: Easy to add new functions - just add `@kernel_function` decorator
5. **Context Aware**: AI can extract employee names and route accordingly

## How It Works

1. **Function Registration**: Functions are decorated with `@kernel_function` and descriptive metadata
2. **AI Analysis**: When a query comes in, the AI analyzes the intent
3. **Automatic Routing**: Based on the function descriptions, AI calls the appropriate function(s)
4. **Response Generation**: AI uses the function results to generate a natural response

## Example Queries

The system can handle various query styles:

### Leave Balance Queries (→ `get_leave_balance`)
- "How many leave days does Alice have left?"
- "What's Bob's current vacation balance?"
- "Check Diana's remaining time off"
- "Alice's leave balance please"

### Policy Queries (→ `query_policy`)
- "How many annual leave days are employees entitled to?"
- "What is the company policy on sick leave?"
- "What are the rules for taking consecutive leave days?"
- "Emergency leave policy"

## Setup and Usage

### Prerequisites
```bash
pip install semantic-kernel>=1.6.0 chromadb sqlalchemy python-dotenv openai
```

### Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Usage

#### Basic Usage
```python
from rag_with_postgres import rag_query

# AI automatically determines this needs get_leave_balance
result = rag_query("How many days does Alice have?")

# AI automatically determines this needs query_policy  
result = rag_query("What's the leave policy?")
```

#### Interactive Mode
```python
import asyncio
from rag_with_postgres import interactive_hr_assistant

asyncio.run(interactive_hr_assistant())
```

#### Testing
```bash
cd leave_tracket
python test_semantic_rag.py
```

## Technical Architecture

### Components

1. **HRPlugin Class**: Contains kernel functions for HR operations
2. **Semantic Kernel**: Handles function calling and routing
3. **Data Sources**: 
   - PostgreSQL for employee leave balances
   - ChromaDB for policy documents
4. **OpenAI Integration**: Provides the AI reasoning for function selection

### Function Definitions

```python
class HRPlugin:
    @kernel_function(
        description="Gets the leave balance for a specific employee from the database. Use this when users ask about remaining leave days, vacation balance, or time off for a specific person."
    )
    def get_leave_balance(self, employee_name: str) -> str:
        # Database query logic
        
    @kernel_function(
        description="Searches company HR policies and procedures. Use this for questions about leave policies, company rules, entitlements, procedures, or general HR information."
    )
    def query_policy(self, query: str) -> str:
        # Vector search logic
```

## Adding New Functions

To add new capabilities, simply add more kernel functions:

```python
@kernel_function(
    description="Submits a new leave request for an employee"
)
def submit_leave_request(
    self, 
    employee_name: str, 
    start_date: str, 
    end_date: str, 
    leave_type: str
) -> str:
    # Implementation
    pass
```

The AI will automatically understand when to call this new function based on the description.

## Backward Compatibility

The original `rag_query()` function is still available and now uses Semantic Kernel internally. Legacy code continues to work without changes.

## Best Practices

1. **Descriptive Function Descriptions**: Make them clear and specific about when to use the function
2. **Type Annotations**: Use proper typing for better AI understanding
3. **Error Handling**: Include try-catch blocks in kernel functions
4. **Testing**: Test with various phrasings to ensure proper routing

## Troubleshooting

### Common Issues

1. **Functions Not Called**: Check if descriptions are clear and specific
2. **Wrong Function Called**: Improve function descriptions to be more distinct
3. **No Response**: Verify OpenAI API key and model availability
4. **Database Errors**: Check DATABASE_URL and database connectivity

### Debug Mode

Set `echo=True` in SQLAlchemy engine creation to see SQL queries:
```python
db = create_engine(DATABASE_URL, echo=True)
```

This intelligent routing system eliminates the need for complex conditional logic and makes the system more maintainable and user-friendly. 