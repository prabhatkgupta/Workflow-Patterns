
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START

load_dotenv()


# ==================== STEP 1: Define State ====================
class ChainingState(TypedDict):
    """State that gets passed between nodes"""
    original_text: str
    summary: str
    translation: str
    current_step: str


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
        "translation": "ji",
        "summary": summary,
        "int": 4
    }


def translate_node(state: str) -> ChainingState:
    """Node 2: Translate the summary to French"""
    print("🌍 Step 2: Translating to French...")
    
    summary = state
    prompt = f"Translate the following summary into French, only return the translation, no other text: {summary}"
    
    response = "Translated version"
    print("in translate", state)
    
    translation = response.strip()
    print(f"Translation: {translation}\n")
    
    return {
        "translation": translation,
        "summary": "Hi",
        "int": 8
    }



# ==================== STEP 3: Build the Graph ====================
def create_workflow():
    """Create and compile the LangGraph workflow"""
    
    # Initialize the graph
    workflow = StateGraph(ChainingState)
    
    # Add nodes
    workflow.add_node("LLM Node 1", summarize_node)
    workflow.add_node("LLM Node 2", translate_node)
    
    # Define edges (flow)
    workflow.set_entry_point("LLM Node 1")
    workflow.add_edge("LLM Node 1", "LLM Node 2")
    workflow.add_edge("LLM Node 2", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app


# ==================== STEP 4: Run the Workflow ====================
def main():
    print("=" * 60)
    print("LANGGRAPH WORKFLOW: SUMMARIZE → TRANSLATE")
    print("=" * 60 + "\n")
    
    # Create the workflow
    app = create_workflow()
    with open("langgraph_chaining.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    
    # Initial state
    initial_state = ChainingState({
        "original_text": "Large language models are powerful AI systems trained on vast amounts of text data. They can generate human-like text, translate languages, write different kinds of creative content, and answer your questions in an informative way.",
        "summary": "",
        "translation": "",
        "current_step": "initial"
    })

    print("type(initial_state)", type(initial_state))
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Original: {initial_state['original_text']}")
    print(f"\nSummary: {initial_state['summary']}")
    print(f"\nTranslation: {final_state['translation']}")
    print(f"\nint: {final_state.get('int', 'NA')}")
    print(f"\nsummary: {final_state['summary']}")
    print(f"\nfinal_state: {final_state}, {type(final_state)}")


if __name__ == "__main__":
    main()