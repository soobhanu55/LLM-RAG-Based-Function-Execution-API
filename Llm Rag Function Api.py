import os
import webbrowser
import fastapi
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
from pydantic import BaseModel
import datetime

# Initialize FastAPI app
app = FastAPI(title="LLM + RAG Function Execution API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FunctionRequest(BaseModel):
    prompt: str

class FunctionResponse(BaseModel):
    function: str
    code: str
    generated_code: str = None
    context: Dict[str, Any] = None

# Context memory (session-based)
context_memory = deque(maxlen=5)  # Store last 5 queries for context

# Predefined function registry with more detailed metadata
functions = {
    "open_chrome": {
        "description": "Open Google Chrome browser",
        "code": "webbrowser.open(\"https://www.google.com\")",
        "parameters": {},
        "category": "application_control",
        "keywords": ["chrome", "browser", "google", "web", "internet", "open"]
    },
    "open_calculator": {
        "description": "Open system calculator",
        "code": "os.system(\"calc\")",
        "parameters": {},
        "category": "application_control",
        "keywords": ["calculator", "calc", "math", "compute", "open"]
    },
    "get_cpu_usage": {
        "description": "Retrieve current CPU usage",
        "code": "os.popen(\"wmic cpu get loadpercentage\").read()",
        "parameters": {},
        "category": "system_monitoring",
        "keywords": ["cpu", "processor", "usage", "load", "system", "monitor"]
    },
    "execute_shell_command": {
        "description": "Execute a shell command safely",
        "code": "os.popen(command).read()",
        "parameters": {"command": "str"},
        "category": "command_execution",
        "keywords": ["shell", "command", "execute", "run", "terminal"]
    }
}

def find_best_function(prompt: str) -> str:
    """Find the most relevant function based on keyword matching"""
    prompt_words = set(prompt.lower().split())
    best_match = None
    best_score = 0
    
    for func_name, func_data in functions.items():
        # Calculate match score based on keywords and description
        keywords = set(func_data["keywords"])
        description_words = set(func_data["description"].lower().split())
        all_words = keywords.union(description_words)
        
        score = len(prompt_words.intersection(all_words))
        
        if score > best_score:
            best_score = score
            best_match = func_name
    
    return best_match if best_score > 0 else None

def generate_dynamic_code(function_name: str, user_prompt: str) -> str:
    """Generate dynamic Python code based on the function and user prompt"""
    function_data = functions[function_name]
    
    # Basic template-based code generation
    code = f"""
try:
    # Execute {function_name}: {function_data['description']}
    {function_data['code']}
    print(f"Successfully executed {function_name}")
except Exception as e:
    print(f"Error executing {function_name}: {{str(e)}}")
"""
    return code

@app.post("/execute", response_model=FunctionResponse)
async def execute_function(request: FunctionRequest):
    prompt = request.prompt
    
    # Find the best matching function
    function_name = find_best_function(prompt)
    
    if not function_name:
        raise HTTPException(status_code=404, detail="No matching function found")
    
    function_data = functions[function_name]
    
    # Generate dynamic code
    generated_code = generate_dynamic_code(function_name, prompt)
    
    # Store context
    context_memory.append({
        "prompt": prompt,
        "function": function_name,
        "timestamp": str(datetime.datetime.now())
    })
    
    try:
        # Execute the function
        exec(function_data['code'])
        
        return FunctionResponse(
            function=function_name,
            code=function_data['code'],
            generated_code=generated_code,
            context={"recent_queries": list(context_memory)}
        )
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing function: {str(e)}")

@app.get("/context")
def get_context():
    """Retrieve context history"""
    return {"context_memory": list(context_memory)}

@app.get("/functions")
def list_functions():
    """List all available functions and their descriptions"""
    return {"functions": functions}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
