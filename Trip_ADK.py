# Travel Recommendation Agent using Google ADK
# Based on Parallelization_ADK.py structure

import os
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

# Set API key (you should set your own key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmPx0NJTx3ln61shGZuqTWqLkVXmeZDSk"

GEMINI_MODEL = "gemini-2.0-flash"
DESTINATION = "tiruchirappalli"  # Change this for different destinations

# --- 1. Define Travel Research Sub-Agents (to run in parallel) ---

# Weather Agent
weather_agent = LlmAgent(
    name="WeatherResearcher",
    model=GEMINI_MODEL,
    instruction=f"""You are a travel weather specialist.
    Based on general knowledge, assess the current weather conditions in {DESTINATION} and determine if it's a good time to travel.
    Provide a clear 1-2 line assessment of whether it's good to travel or not.
    Output *only* the weather assessment.
    """,
    description=f"Assesses weather conditions in {DESTINATION}.",
    output_key="weather_result",
)

# Flights Agent
flights_agent = LlmAgent(
    name="FlightsResearcher",
    model=GEMINI_MODEL,
    instruction=f"""You are a flight booking specialist.
    Based on general knowledge, provide typical flight options from Tamil Nadu to {DESTINATION}.
    Provide details on: number of flights per day, typical departure/arrival times, and travel duration.
    Output *only* the flight information.
    """,
    description=f"Provides flight options to {DESTINATION}.",
    output_key="flights_result",
)

# Hotels Agent
hotels_agent = LlmAgent(
    name="HotelsResearcher",
    model=GEMINI_MODEL,
    instruction=f"""You are an accommodation specialist.
    Based on general knowledge, list typical budget hostels in {DESTINATION}.
    List only hostel names with their locations and per-day prices.
    Output *only* the hostel list.
    """,
    description=f"Lists hostels in {DESTINATION}.",
    output_key="hotels_result",
)

# Cost Estimation Agent
cost_agent = LlmAgent(
    name="CostEstimator",
    model=GEMINI_MODEL,
    instruction=f"""You are a travel cost analyst.
    Based on general knowledge, estimate typical costs for a trip to {DESTINATION} including travel, accommodation, food, and local transport.
    Provide only the total estimated cost range.
    Output *only* the cost range.
    """,
    description=f"Estimates trip costs to {DESTINATION}.",
    output_key="cost_result",
)

# --- 2. Create Parallel execution ---

# --- 3. Define the Merger Agent ---
merger_agent = LlmAgent(
    name="TravelSynthesisAgent",
    model=GEMINI_MODEL,
    instruction=f"""You are a travel planning assistant responsible for combining research findings into a comprehensive travel recommendation.

    **Input Information:**
    * **Weather:** {{weather_result}}
    * **Flights:** {{flights_result}}
    * **Hotels:** {{hotels_result}}
    * **Cost:** {{cost_result}}

    **Output Format:**
    ## Travel Recommendation for {DESTINATION}

    ### Weather & Best Time to Travel
    {{weather_result}}

    ### Flight Options from Tamil Nadu
    {{flights_result}}

    ### Recommended Hostels
    {{hotels_result}}

    ### Estimated Total Cost
    {{cost_result}}

    Output *only* the structured recommendation following this format.
    """,
    description="Combines travel research findings into a structured recommendation.",
)

# --- 4. Create the SequentialAgent (Runs agents one by one) ---
sequential_travel_agent = SequentialAgent(
    name="TravelPlanningPipeline",
    sub_agents=[weather_agent, flights_agent, hotels_agent, cost_agent, merger_agent],
    description="Coordinates sequential travel research and synthesizes the results.",
)

root_agent = sequential_travel_agent

if __name__ == "__main__":
    import asyncio
    import sys
    import os
    from contextlib import redirect_stderr
    from io import StringIO
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    import uuid

    async def run_travel_planning():
        runner = InMemoryRunner(root_agent, app_name="agents")

        user_id = "user_123"
        session_id = str(uuid.uuid4())

        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        print(f"Starting travel planning research for {DESTINATION}...")

        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=f"Plan a trip to {DESTINATION}")],
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, "text") and event.content.text:
                    result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    result = "".join(text_parts)

                print("\n" + "="*60)
                print(f"TRAVEL RECOMMENDATION FOR {DESTINATION.upper()}")
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
            asyncio.run(run_travel_planning())
            print("\n✅ Travel planning completed successfully")
        except KeyboardInterrupt:
            print("\n❌ Execution interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Wait a bit for threads to finish printing errors
            time.sleep(10)
            # Restore stderr
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)
            os.close(devnull)

    main()