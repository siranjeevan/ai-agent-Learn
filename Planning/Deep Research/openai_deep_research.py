#python Planning/Deep Research/openai_deep_research.py


import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the client with your API key
# Ensure OPENAI_API_KEY is set in your .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the agent's role and the user's research question
system_message = """You are a professional researcher preparing
a structured, data-driven report.
Focus on data-rich insights, use reliable sources, and include
inline citations."""

user_query = "Research the economic impact of semaglutide on global healthcare systems."

print(f"Starting research on: {user_query}")

try:
    # Create the Deep Research API call
    # Note: 'o3-deep-research-2025-06-26' and 'client.responses' are hypothetical/preview references
    # Ensure your SDK version supports this or adjust to standard client.chat.completions.create
    response = client.responses.create(
        model="o3-deep-research-2025-06-26",
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_message
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_query
                    }
                ]
            }
        ],
        # reasoning={"summary": "auto"},
        tools=[{"type": "web_search_preview"}]
    )

    # Access and print the final report
    # The output structure here assumes the specific response format of the O3 preview
    final_report = response.output[-1].content[0].text
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(final_report)

    # --- ACCESS INLINE CITATIONS AND METADATA ---
    print("\n--- CITATIONS ---")
    annotations = response.output[-1].content[0].annotations

    if not annotations:
        print("No annotations found in the report.")
    else:
        for i, citation in enumerate(annotations):
            cited_text = final_report[citation.start_index:citation.end_index]
            print(f"Citation {i + 1}:")
            print(f"  Cited Text: {cited_text}")
            print(f"  Title: {citation.title}")
            print(f"  URL: {citation.url}")
            print(f"  Location: chars {citation.start_index}–{citation.end_index}")
            print("\n" + "-" * 30 + "\n")

    # --- INSPECT INTERMEDIATE STEPS ---
    print("\n--- INTERMEDIATE STEPS ---")

    # 1. Reasoning Steps
    try:
        reasoning_step = next(item for item in response.output if item.type == "reasoning")
        print("\n[Found a Reasoning Step]")
        for summary_part in reasoning_step.summary:
            print(f" - {summary_part.text}")
    except StopIteration:
        print("\nNo reasoning steps found.")

    # 2. Web Search Calls
    try:
        search_step = next(item for item in response.output if item.type == "web_search_call")
        print("\n[Found a Web Search Call]")
        print(f" Query Executed: '{search_step.action['query']}'")
        print(f" Status: {search_step.status}")
    except StopIteration:
        print("\nNo web search steps found.")

    # 3. Code Execution
    try:
        code_step = next(item for item in response.output if item.type == "code_interpreter_call")
        print("\n[Found a Code Execution Step]")
        print(" Code Input:")
        print(f" ```python\n{code_step.input}\n ```")
        print(" Code Output:")
        print(f" {code_step.output}")
    except StopIteration:
        print("\nNo code execution steps found.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Note: This script uses valid syntax for a specific model version ('o3-deep-research').")
    print("Please ensure your OpenAI SDK is updated and you have access to this model preview.")
