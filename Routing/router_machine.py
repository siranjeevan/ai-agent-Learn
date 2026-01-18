# Copyright (c) 2025 Marco Fago
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.
import asyncio
import os
import uuid
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event

def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the booking was handled.
    """
    print("------------- Booking Handler Called -------------")
    return f"Booking action for '{request}' has been simulated."


def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's question.
    Returns:
        A message indicating the information request was handled.
    """
    print("------------- Info Handler Called ----------------")
    return (
        f"Information request for '{request}'. "
        f"Result: Simulated information retrieval."
    )


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""

    print("------------- UnClear Handler Called ----------------")
    return (
        f"Coordinator could not delegate request: '{request}'. "
        f"Please clarify."
    )

booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)
unclear_tool = FunctionTool(unclear_handler)

booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description=(
        "A specialized agent that handles all flight and hotel booking "
        "requests by calling the booking tool."
    ),
    tools=[FunctionTool(booking_handler)],
)

info_agent = Agent(
    name="Info",
    model="gemini-2.0-flash",
    description=(
        "A specialized agent that provides general information and answers "
        "user questions by calling the info tool."
    ),
    tools=[info_tool],
)
unclear_agent = Agent(
    name="Unclear",
    model="gemini-2.0-flash",
    description=(
        "A specialized agent that handles unclear requests by calling the unclear tool."
    ),
    tools=[FunctionTool(unclear_handler)],
)

# Define the parent agent with explicit delegation instructions
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyze incoming user requests "
        "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"

        "- Delegate to the 'Booker' agent ONLY if the request is strictly about booking flights or hotels.\n"

        "- Delegate to the 'Info' agent ONLY if the request asks for external, factual, real-world information "
        "such as places, definitions, explanations, guides, prices, or general knowledge.\n"

        "- Questions about the system, the agent, identity, greetings, personal opinions, or anything that is "
        "not clearly booking or factual information MUST be delegated to the 'Unclear' agent.\n"
    ),
    description=(
        "A coordinator that routes user requests to the correct "
        "specialist agent."
    ),
    sub_agents=[booking_agent, info_agent, unclear_agent],
)



# --- Execution Logic ---
async def run_coordinator(runner: InMemoryRunner, request: str) -> str:
    """Runs the coordinator agent with a given request and delegates."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())

        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=request)],
            ),
        ):
            if event.is_final_response() and event.content:
                # Try to get text directly from event.content to avoid iterating parts
                if hasattr(event.content, "text") and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    # Fallback: Iterate through parts and extract text
                    text_parts = [
                        part.text for part in event.content.parts if part.text
                    ]
                    final_result = "".join(text_parts)
                # Assuming the loop should break after the final response
                break

        print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


async def main():
    """Main function to run the ADK example."""
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    # Create ONE runner and reuse it for all requests
    runner = InMemoryRunner(coordinator)


    result_a = await run_coordinator(runner, "Book me a hotel in Paris.")
    print(f"Final Output A: {result_a}")

    result_b = await run_coordinator(
        runner, "What is the highest mountain in the world?"
    )
    print(f"Final Output B: {result_b}")

    result_c = await run_coordinator(
        runner, "Tell me a random fact."
    )  # Should go to Info
    print(f"Final Output C: {result_c}")

    result_d = await run_coordinator(
        runner, "Find flights to Tokyo next month."
    )  # Should go to Booker
    print(f"Final Output D: {result_d}")

    result_e = await run_coordinator(
        runner, "What is Your Name "
    )  # Should go to Booker
    print(f"Final Output E: {result_e}")

if __name__ == "__main__":

    asyncio.run(main())