"""
LangGraph Implementation - Multi-Step AI Workflow
Converts sequential Gemini calls into a LangGraph state machine
"""

import torch
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from operator import add

load_dotenv()


# ==================== STEP 1: Define State ====================
class ParallelState(TypedDict):
    """State that gets passed between nodes"""
    original_text: str
    summary: bool
    translation: bool
    decider: float
    result_text: Annotated[str, add]




def greet_hello_node(state: ParallelState) -> ParallelState:
    print("Hello from greet_hello_node")
    print(f"Here is my workflow state {state}")

    return {
        "result_text": "This is"
    }



def decision_maker(state: ParallelState) -> str:
    print("I'm inside a decision maker state")
    if state["decider"] > 0.5:
        return "summarize"
    return "translate"



# ==================== STEP 2: Define Node Functions ====================
def summarize_node(state: str) -> str:
    """Node 1: Summarize the original text"""
    print("📝 Step 1: Summarizing text...")
    
    original_text = state
    prompt = f"Summarize the following text in one sentence: {original_text}"
    
    response = "Summarized version"
    print("in summarize_node", state)
    
    summary = response.strip()
    print(f"Summary: {summary}\n")
    
    return {
        "summary": True,
        "result_text": "Not acceptable"
    }


def translate_node(state: str) -> ParallelState:
    """Node 2: Translate the summary to French"""
    print("🌍 Step 2: Translating to French...")
    
    summary = state
    prompt = f"Translate the following summary into French, only return the translation, no other text: {summary}"
    
    response = "Translated version"
    print("in translate", state)
    
    translation = response.strip()
    print(f"Translation: {translation}\n")
    
    return {
        "translation": True,
        "result_text": "Not good"
    }


def print_hello_world(state: ParallelState) -> ParallelState:
    print("Hello world man")
    print("state['result_text']", state["result_text"])




# ==================== STEP 3: Build the Graph ====================
def create_workflow():
    """Create and compile the LangGraph workflow"""
    
    # Initialize the graph
    workflow = StateGraph(ParallelState)
    
    # Add nodes
    workflow.add_node("Distributor Node", greet_hello_node)
    workflow.add_node("Parallel Node 1", summarize_node)
    workflow.add_node("Parallel Node 2", translate_node)
    workflow.add_node("Synthesizer Node", print_hello_world)

    # Define edges (flow)
    workflow.set_entry_point("Distributor Node")
    workflow.add_edge("Distributor Node", "Parallel Node 1")
    workflow.add_edge("Distributor Node", "Parallel Node 2")
    workflow.add_edge("Parallel Node 1", "Synthesizer Node")
    workflow.add_edge("Parallel Node 2", "Synthesizer Node")
    workflow.add_edge("Synthesizer Node", END)


    
    # Compile the graph
    app = workflow.compile()
    
    return app


# ==================== STEP 4: Run the Workflow ====================
def main():
    print("=" * 60 + "\n")
    
    # Create the workflow
    app = create_workflow()
    with open("langgraph_parallelization.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    
    # Initial state
    initial_state = ParallelState({
        "original_text": "Large language models are powerful AI systems trained on vast amounts of text data. They can generate human-like text, translate languages, write different kinds of creative content, and answer your questions in an informative way.",
        "summary": False,
        "translation": False,
        "decider": torch.randn(1).item(),
        "result_text": ""
    })

    print("type(initial_state)", type(initial_state))
    print(f"Original: {initial_state['original_text']}")
    print(f"\nSummary: {initial_state['summary']}")
    print(f"\nTranslation: {initial_state['translation']}")

    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nint: {final_state.get('int', 'NA')}")
    print(f"\nsummary: {final_state['summary']}")
    print(f"\nTranslation: {final_state['translation']}")
    print(f"\nfinal_state: {final_state}, {type(final_state)}")


if __name__ == "__main__":
    main()