import os
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from typing import List

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

load_dotenv()

# structured data for the two-step process
class State(TypedDict):
    """State container for the tourist information agent.
    
    Attributes:
        messages: Conversation history with proper message handling[citation:2]
        destinations: List of country-city pairs for iconic tour destinations
        attractions: Dictionary mapping cities to their famous attractions
        step: Current step in the process (1 or 2)
    """
    messages: Annotated[list, add_messages]
    destinations: List[str]
    attractions: dict
    step: int

# Initialize the Language Model (LLM)
try:
    # Using gemma3 via ollama - local run - README.md shows the ollama implementation
    llm = ChatOllama(model="gemma3")
except Exception as e:
    print(f"Error initializing Ollama model: {e}")
    print("Please ensure Ollama is installed and the 'gemma3' model is available.")
    exit()

# Define the System Prompts
GET_DESTINATIONS_PROMPT = SystemMessage(content="""You are a travel expert. Generate 10 iconic tour destinations in the format 'country-city' based on the most popular travel destinations from 2017-2024. 
For example: 'Sri Lanka-Weligama', 'Japan-Tokyo', 'Italy-Rome', 'France-Paris'. 
Return ONLY a list of 10 country-city pairs, one per line, no additional text.""")

GET_ATTRACTIONS_PROMPT_TEMPLATE = SystemMessage(content="""You are a local guide. For the city {city}, list 5 of the most famous things to watch or visit.
Return ONLY a numbered list of 5 attractions, one per line, no additional text.
Example format:
1. Eiffel Tower
2. Louvre Museum
3. Notre-Dame Cathedral
4. Champs-Élysées
5. Montmartre""")

# Define Node Functions for the Two-Step Process

def get_destinations_node(state: State) -> dict:
    """
    Step 1: Get iconic tour destinations country-city wise.
    This node queries the LLM for 10 iconic destinations.
    """
    print("Step 1: Getting iconic tour destinations...")
    
    # Query the LLM for destinations
    response = llm.invoke([GET_DESTINATIONS_PROMPT])
    
    # Parse the response to extract destinations
    destinations_text = response.content.strip()
    destinations = [line.strip() for line in destinations_text.split('\n') if line.strip()]
    
    # Limit to 10 destinations if more were returned
    destinations = destinations[:10]
    
    print(f"Found {len(destinations)} destinations: {destinations}")
    
    return {
        "destinations": destinations,
        "step": 2,  
        "messages": [response] 
    }

def get_attractions_node(state: State) -> dict:
    """
    Step 2: For each destination, get famous attractions.
    This processes all destinations from step 1.
    """
    print("\nStep 2: Getting famous attractions for each destination...")
    
    attractions = {}
    
    for destination in state["destinations"]:
        try:
            # Parse country and city from the destination string
            if '-' in destination:
                country, city = destination.split('-', 1)
            else:
                # Fallback if format is different
                country = "Unknown"
                city = destination
            
            print(f"\nProcessing {city}, {country}...")
            
            # Create city-specific prompt
            attractions_prompt = GET_ATTRACTIONS_PROMPT_TEMPLATE.model_copy()
            attractions_prompt.content = GET_ATTRACTIONS_PROMPT_TEMPLATE.content.format(city=city)
            
            # Query the LLM for attractions
            response = llm.invoke([attractions_prompt])
            
            # Parse the response
            attractions_text = response.content.strip()
            attraction_list = [line.strip() for line in attractions_text.split('\n') 
                             if line.strip() and any(char.isdigit() or char.isalpha() for char in line)]
            
            # Clean up the list (remove numbers and dots)
            clean_attractions = []
            for item in attraction_list:
                # Remove leading numbers and punctuation
                clean_item = item.lstrip('0123456789. ')
                if clean_item:
                    clean_attractions.append(clean_item)
            
            # Store attractions for this city
            attractions[destination] = clean_attractions[:5]  # Limit to 5
            
            print(f"  Found {len(clean_attractions)} attractions for {city}")
            
        except Exception as e:
            print(f"  Error processing {destination}: {e}")
            attractions[destination] = ["Error retrieving attractions"]
    
    return {
        "attractions": attractions,
        "step": 3,  # Mark process as complete
        "messages": []  # No additional messages needed
    }

def display_results_node(state: State) -> dict:
    """
    Final node: Display all the collected information.
    """
    print("TOURIST INFORMATION AGENT - FINAL RESULTS")
    print("-----------------------------------------")
    
    print("\nICONIC TOUR DESTINATIONS (Country-City):")
    for i, destination in enumerate(state["destinations"], 1):
        print(f"  {i}. {destination}")
    
    print("\nFAMOUS ATTRACTIONS FOR EACH CITY:")
    for destination, attraction_list in state["attractions"].items():
        country_city = destination.split('-')
        if len(country_city) == 2:
            country, city = country_city
            print(f"\n  {city}, {country}:")
        else:
            print(f"\n  {destination}:")
        
        for j, attraction in enumerate(attraction_list, 1):
            print(f"    {j}. {attraction}")
    
    print("\n" + "-------------------------------------------------")
    print("Process completed successfully!")
    print("------------------------------------------------")
    
    return {"step": 0}  # Reset step for potential reuse

# --- 5. Define Conditional Edge Logic ---
def route_by_step(state: State) -> str:
    """
    Conditional routing based on the current step.
    This controls the flow through the 2-step process.
    """
    step = state.get("step", 1)
    
    if step == 1:
        return "get_destinations"
    elif step == 2:
        return "get_attractions"
    elif step == 3:
        return "display_results"
    else:
        return END

# --- 6. Build the LangGraph Workflow ---
graph_builder = StateGraph(State)

# Add nodes for each step[citation:5]
graph_builder.add_node("get_destinations", get_destinations_node)
graph_builder.add_node("get_attractions", get_attractions_node)
graph_builder.add_node("display_results", display_results_node)

# Set the entry point
graph_builder.set_entry_point("get_destinations")

# Add conditional edges to control the flow[citation:1]
graph_builder.add_conditional_edges(
    "get_destinations",
    route_by_step,
    {
        "get_attractions": "get_attractions",
        "display_results": "display_results",
        END: END
    }
)

graph_builder.add_conditional_edges(
    "get_attractions",
    route_by_step,
    {
        "display_results": "display_results",
        END: END
    }
)

graph_builder.add_edge("display_results", END)

# Compile the graph into an executable workflow[citation:2]
app = graph_builder.compile()

# # Display the graph structure
# try:
#     from IPython.display import Image, display
#     display(Image(app.get_graph().draw_mermaid_png()))
# except:
#     print("Graph visualization requires IPython and graphviz")

# --- 7. Run the Automated Agent ---
print("-------------------------------------")
print("AUTOMATED TOURIST INFORMATION AGENT")
print("-------------------------------------")
print("This agent will automatically:")
print("1. Get 10 iconic tour destinations (country-city)")
print("2. Find famous attractions for each city")
print("-------------------------------------")

# Initialize the state with step 1
initial_state = {
    "messages": [],
    "destinations": [],
    "attractions": {},
    "step": 1
}

# Run the complete workflow automatically[citation:5]
try:
    print("\nStarting automated process...")
    
    # Stream through the execution
    for output in app.stream(initial_state):
        for node_name, node_output in output.items():
            if node_name != "__end__":
                # The nodes already print their progress
                pass
    
    print("-------------------------------------")
    print("Agent execution completed!")
    print("-------------------------------------")
    
except KeyboardInterrupt:
    print("\n\nProcess interrupted by user.")
except Exception as e:
    print(f"\nError during execution: {e}")
    print("Please check your Ollama installation and try again.")