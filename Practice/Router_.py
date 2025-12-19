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

def cancel_order_handler(request: str) -> str:
    """Handles order cancellation requests."""
    print("\n--- DELEGATING TO CANCEL ORDER HANDLER ---")
    return (
        f"Cancel Order Handler processed request: '{request}'. "
        f"Result: Order cancellation workflow initiated."
    )

def change_delivery_address_handler(request: str) -> str:
    """Handles delivery address change requests."""
    print("\n--- DELEGATING TO CHANGE DELIVERY ADDRESS HANDLER ---")
    return (
        f"Change Delivery Address Handler processed request: '{request}'. "
        f"Result: Delivery address change tool activated."
    )

def track_order_handler(request: str) -> str:
    """Handles order tracking requests."""
    print("\n--- DELEGATING TO TRACK ORDER HANDLER ---")
    return (
        f"Track Order Handler processed request: '{request}'. "
        f"Result: Order tracking tool executed."
    )

def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return (
        f"Coordinator could not delegate request: '{request}'. "
        f"Please clarify your e-commerce order request."
    )

def classify_request(request: str) -> str:
    """Classify the user's request using the LLM."""
    prompt = f"""Analyze the user's e-commerce order request and determine which specialist handler should process it.
- If the request is related to canceling an order, output 'cancel'.
- If the request is related to changing delivery address, output 'address'.
- If the request is related to tracking an order, output 'track'.
- If the request is unclear or doesn't fit any category, output 'unclear'.
ONLY output one word: 'cancel', 'address', 'track', or 'unclear'.

User request: {request}"""

    if not llm_available:
        return "unclear"

    try:
        response = model.generate_content(prompt)
        decision = response.text.strip().lower()
        # Validate the response
        if decision in ["cancel", "address", "track", "unclear"]:
            return decision
        else:
            return "unclear"
    except Exception as e:
        print(f"Error classifying request: {e}")
        return "unclear"

def route_request(request: str) -> str:
    """Route the request to the appropriate handler."""
    decision = classify_request(request)

    if decision == "cancel":
        return cancel_order_handler(request)
    elif decision == "address":
        return change_delivery_address_handler(request)
    elif decision == "track":
        return track_order_handler(request)
    else:
        return unclear_handler(request)

def main():
    if not llm_available:
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print("=== E-Commerce Order Router ===")
    print("Enter your order requests (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Your request: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter a request.\n")
            continue
            
        result = route_request(user_input)
        print(f"Response: {result}\n")


if __name__ == "__main__":
    main()
