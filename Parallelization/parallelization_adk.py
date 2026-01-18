# Copyright (c) 2025 Marco Fago
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.
# python Parallelization_ADK.py 


import os
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmPx0NJTx3ln61shGZuqTWqLkVXmeZDSk"

GEMINI_MODEL = "gemini-2.0-flash"

# --- 1. Define Researcher Sub-Agents (to run in parallel) ---

# Researcher 1: Renewable Energy
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
    description="Researches renewable energy sources.",
    tools=[google_search],
    output_key="renewable_energy_result",
)

# Researcher 2: Electric Vehicles
researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
    description="Researches electric vehicle technology.",
    tools=[google_search],
    output_key="ev_technology_result",
)

# Researcher 3: Carbon Capture
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in climate solutions.
    Research the current state of 'carbon capture methods'.
    Use the Google Search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *only* the summary.
    """,
    description="Researches carbon capture methods.",
    tools=[google_search],
    output_key="carbon_capture_result",
)

# --- 2. Create the ParallelAgent (Runs researchers concurrently) ---
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Runs multiple research agents in parallel to gather information.",
)

# --- 3. Define the Merger Agent (Runs *after* the parallel agents) ---
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""You are an AI Assistant responsible for combining research findings into a structured report.
    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas.
    Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.
    
    **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. 
    Do NOT add any external knowledge, facts, or details not present in these specific summaries.**

    **Input Summaries:**
    * **Renewable Energy:** {renewable_energy_result}
    * **Electric Vehicles:** {ev_technology_result}
    * **Carbon Capture:** {carbon_capture_result}
    
    **Output Format:**
    ## Summary of Recent Sustainable Technology Advancements

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

    ### Electric Vehicle Findings
    (Based on EVResearcher's findings)
    [Synthesize and elaborate *only* on the EV input summary provided above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

    Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, 
    and strictly adhere to using only the provided input summary content.
    """,
    description=(
        "Combines research findings from parallel agents into a structured, "
        "cited report, strictly grounded on provided inputs."
    ),
)

# --- 4. Create the SequentialAgent (Orchestrates the overall flow) ---
sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesizes the results.",
)

root_agent = sequential_pipeline_agent

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    from contextlib import redirect_stderr
    from io import StringIO
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    import uuid
    
    async def run_research():
        runner = InMemoryRunner(root_agent, app_name="agents")
        
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        
        print("Starting parallel research and synthesis...")
        
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Research sustainable technology advancements")],
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, "text") and event.content.text:
                    result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    result = "".join(text_parts)
                
                print("\n" + "="*60)
                print("FINAL RESEARCH SYNTHESIS REPORT")
                print("="*60)
                print(result)
                print("="*60)
                break
    
    def main():
        import os
        import time

        # Save original stderr
        original_stderr_fd = os.dup(2)
        # Redirect stderr to /dev/null to suppress cleanup errors
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, 2)

        try:
            asyncio.run(run_research())
            print("\n✅ Execution completed successfully")
        except KeyboardInterrupt:
            print("\n❌ Execution interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Wait a bit for threads to finish printing errors
            time.sleep(3)
            # Restore stderr
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)
            os.close(devnull)
    
    main()
