import os
from typing_extensions import TypedDict, Annotated
from typing import List

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- 1. Define the Agent State ---
# The state is a TypedDict that holds the conversation history.
# 'Annotated' with 'add_messages' automatically appends new messages to the list.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- 2. Initialize the Language Model (LLM) ---
# Ensure Ollama is running in your terminal (ollama run gemma3)
# or just "ollama serve" in a background process.
# We use ChatOllama from langchain-ollama to interface with the local model.
try:
    llm = ChatOllama(model="gemma3")
except Exception as e:
    print(f"Error initializing Ollama model: {e}")
    print("Please ensure Ollama is installed and the 'gemma3' model is available.")
    exit()

# --- 3. Define the Node Function ---
# This function takes the current state, invokes the LLM, and returns the updated state.
def chatbot(state: State):
    """
    Invokes the LLM to generate a response based on the conversation history.
    """
    # The 'messages' key from the state is passed to the model
    result = llm.invoke(state["messages"])
    # The new message from the LLM is added to the messages list in the state
    return {"messages": [result]}

# --- 4. Build the LangGraph Workflow ---

# Create a StateGraph instance with our defined state schema
graph_builder = StateGraph(State)

# Add the 'chatbot' function as a node named 'llm_node'
graph_builder.add_node("llm_node", chatbot)

# Set the entry point for the graph (where the process starts)
graph_builder.set_entry_point("llm_node")

# Define the flow: after 'llm_node' runs, the process ends (for a simple chatbot)
graph_builder.add_edge("llm_node", END)

# Compile the graph into an executable workflow
app = graph_builder.compile()

# Display the graph
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# --- 5. Run the Chatbot ---

print("Chatbot initialized with Gemma3 via Ollama. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    # Invoke the compiled graph with the user's message as the initial state
    # LangGraph handles the state management behind the scenes
    inputs = {"messages": [("human", user_input)]}
    
    # Iterate through the graph execution (for a simple graph, this is one step)
    for output in app.stream(inputs):
        # Stream the output content as it's generated (optional, can also use app.invoke)
        for key, value in output.items():
            if key != "__end__":
                print(f"Bot: {value['messages'][-1].content}")
