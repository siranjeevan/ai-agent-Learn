import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

load_dotenv()

# --- Configuration ---
DESTINATION = "Antarctica"  # Change this variable to plan for different destinations

try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# Optional: Add retry logic for rate limits
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def invoke_with_retry(chain, input_data):
    return await chain.ainvoke(input_data)


weather_chain: Runnable = (
    ChatPromptTemplate.from_messages(
        [
            ("system", f"Analyze the weather in {DESTINATION} and tell me clearly in one or two lines whether it is a good time to travel or not. No extra details."),
            ("user", f"Check weather and best time to travel to {DESTINATION}"),
        ]
    )
    | llm
    | StrOutputParser()
)

flights_chain: Runnable = (
    ChatPromptTemplate.from_messages(
        [
            ("system", f"Give only the flight details from Tamil Nadu to {DESTINATION}: number of flights per day, departure times, arrival times, and total travel duration. No extra explanation."),
            ("user", f"Find flights from Tamil Nadu to {DESTINATION}"),
        ]
    )
    | llm
    | StrOutputParser()
)

hotels_chain: Runnable = (
    ChatPromptTemplate.from_messages(
        [
            ("system", f"List only hostel names in {DESTINATION} with location and per-day price. No description. No extra text."),
            ("user", f"Find hotels in {DESTINATION}"),
        ]
    )
    | llm
    | StrOutputParser()
)

cost_chain: Runnable = (
    ChatPromptTemplate.from_messages(
        [
            ("system", f"Estimate the total trip cost to {DESTINATION} including travel, stay, food, and local transport. Give only the final total range. No explanation."),
            ("user", f"Estimate total trip cost to {DESTINATION}"),
        ]
    )
    | llm
    | StrOutputParser()
)

# --- Build the Parallel Chain ---
travel_chain = RunnableParallel(
    {
        "weather": weather_chain,
        "flights": flights_chain,
        "hotels": hotels_chain,
        "cost": cost_chain,
    }
)

# --- Run the Travel Agent ---
async def run_travel_agent() -> None:
    """
    Asynchronously invokes the parallel travel recommendation chains
    and prints the results.
    """
    if not llm:
        print("LLM not initialized. Cannot run travel agent.")
        return

    print(f"\n--- Travel Recommendation Agent for {DESTINATION} ---")
    try:
        response = await invoke_with_retry(travel_chain, {})
        print("\n" + "="*50)
        print(f"TRAVEL RECOMMENDATION RESULTS FOR {DESTINATION.upper()}")
        print("="*50)
        print(f"\n🌤️  WEATHER & BEST TIME TO TRAVEL TO {DESTINATION.upper()}:")
        print("-" * 40)
        print(response['weather'])
        print(f"\n✈️  FLIGHTS FROM TAMIL NADU TO {DESTINATION.upper()}:")
        print("-" * 40)
        print(response['flights'])
        print(f"\n🏨  HOSTELS IN {DESTINATION.upper()}:")
        print("-" * 40)
        print(response['hotels'])
        print(f"\n💰  ESTIMATED TOTAL TRIP COST TO {DESTINATION.upper()}:")
        print("-" * 40)
        print(response['cost'])
        print("="*50)
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(run_travel_agent())
