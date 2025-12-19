import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Language model initialized: gemini-2.5-flash")
    llm_available = True
except Exception as e:
    print(f"Error initializing language model: {e}")
    model = None
    llm_available = False


def get_weather_info(destination: str) -> str:
    """Get weather and best time to travel information."""
    prompt = f"Analyze the weather in {destination} and tell me clearly in one or two lines whether it is a good time to travel or not. No extra details."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error getting weather info: {e}"

def get_flight_info(destination: str) -> str:
    """Get flight information."""
    prompt = f"Give only the flight details from Tamil Nadu to {destination}: number of flights per day, departure times, arrival times, and total travel duration. No extra explanation."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error getting flight info: {e}"

def get_hotel_info(destination: str) -> str:
    """Get hotel information."""
    prompt = f"List only hostel names in {destination} with location and per-day price. No description. No extra text."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error getting hotel info: {e}"

def get_cost_info(destination: str) -> str:
    """Get cost estimation."""
    prompt = f"Estimate the total trip cost to {destination} including travel, stay, food, and local transport. Give only the final total range. No explanation."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error getting cost info: {e}"

def run_travel_agent(destination: str) -> None:
    """Generate travel recommendations for the given destination."""
    if not llm_available:
        print("LLM not initialized. Cannot run travel agent.")
        return

    print(f"\n--- Travel Recommendation Agent for {destination} ---")
    print("\n" + "="*50)
    print(f"TRAVEL RECOMMENDATION RESULTS FOR {destination.upper()}")
    print("="*50)
    
    print(f"\n🌤️  WEATHER & BEST TIME TO TRAVEL TO {destination.upper()}:")
    print("-" * 40)
    print(get_weather_info(destination))
    
    print(f"\n✈️  FLIGHTS FROM TAMIL NADU TO {destination.upper()}:")
    print("-" * 40)
    print(get_flight_info(destination))
    
    print(f"\n🏨  HOSTELS IN {destination.upper()}:")
    print("-" * 40)
    print(get_hotel_info(destination))
    
    print(f"\n💰  ESTIMATED TOTAL TRIP COST TO {destination.upper()}:")
    print("-" * 40)
    print(get_cost_info(destination))
    print("="*50)

def main():
    if not llm_available:
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print("=== Travel Recommendation Agent ===")
    print("Enter destinations to get travel recommendations (type 'quit' to exit):\n")
    
    while True:
        destination = input("Enter destination: ").strip()
        
        if destination.lower() in ['quit', 'exit', 'q']:
            print("Safe travels!")
            break
            
        if not destination:
            print("Please enter a destination.\n")
            continue
            
        run_travel_agent(destination)
        print("\n")

if __name__ == "__main__":
    main()
