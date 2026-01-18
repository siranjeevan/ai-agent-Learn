import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

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

def safe_generate(prompt, delay=1):
    """Generate content with rate limiting."""
    try:
        time.sleep(delay)  # Rate limiting
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            print("Rate limit reached. Please wait and try again later.")
            return "Rate limit exceeded"
        return f"Error: {e}"


def check_product(text_input):
    prompt = (
        "You are a strict validator. Read the input data and check if it is "
        "factually correct. Reply with only one word: Yes or No. "
        "Do not explain. Do not add any extra text.\n\n"
        f"{text_input}"
    )
    return safe_generate(prompt)


def get_product_details(text_input):
    prompt = (
        "Extract the correct product details for this product in bullet format:\n"
        f"Product: {text_input}\n"
        "Provide the accurate specifications:"
    )
    return safe_generate(prompt)


def message_passing(wrong_data, correct_data):
    prompt = (
        "You are a data validation assistant. Compare the user's product data "
        "with the correct reference data. If any mismatch is found:\n"
        "- Display a clear warning message.\n"
        "- Show the wrong and correct values in a table.\n"
        "- Ask the user to update the product details.\n\n"
        f"Wrong Data:\n{wrong_data}\n\n"
        f"Correct Data:\n{correct_data}"
    )
    return safe_generate(prompt)


def run_pipeline(user_input):
    if not llm_available:
        print("LLM not available")
        return
        
    print("User Input:")
    print(user_input)
    print("-" * 60)

    validation_result = check_product(user_input)
    if "Rate limit" in validation_result or "Error" in validation_result:
        print(validation_result)
        return
        
    print(f"Validation Result: {validation_result}")
    print("-" * 60)

    if validation_result.lower() == "yes":
        print("✅ Data seems correct.")
    else:
        print("⚠️ Data seems incorrect. Preparing comparison...")
        product_details = get_product_details(user_input)
        
        if "Rate limit" in product_details or "Error" in product_details:
            print(product_details)
            return
            
        message_passing_result = message_passing(user_input, product_details)
        print(message_passing_result)
        print("-" * 60)

def main():
    if not llm_available:
        print("Skipping execution due to LLM initialization failure.")
        return
    
    print("=== Product Data Validation Pipeline ===")
    print("Enter product details for validation (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter product details: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter product details.\n")
            continue
            
        run_pipeline(user_input)
        print("\n")

if __name__ == "__main__":
    main()
