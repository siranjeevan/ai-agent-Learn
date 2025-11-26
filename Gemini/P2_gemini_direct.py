import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def check_product(text_input):
    prompt = (
        "You are a strict validator. Read the input data and check if it is "
        "factually correct. Reply with only one word: Yes or No. "
        "Do not explain. Do not add any extra text.\n\n"
        f"{text_input}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def get_product_details(text_input):
    prompt = (
        "Extract the correct product details for this product in bullet format:\n"
    "Product: "f"{user_input}"
    "Provide the accurate specifications:"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


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
    response = model.generate_content(prompt)
    return response.text.strip()


def run_pipeline(user_input):
    print("User Input:")
    print(user_input)
    print("-" * 60)

    validation_result = check_product(user_input)
    print(f"Validation Result: {validation_result}")
    print("-" * 60)

    if validation_result.lower() == "yes":
        print("✅ Data seems correct. Important details:")

    else:
        print("⚠️ Data seems incorrect. Preparing comparison...")
        product_details = get_product_details(user_input)

        message_passing_result = message_passing(user_input, product_details)
        print(message_passing_result)
        print("-" * 60)

if __name__ == "__main__":
    user_input = "Product: I Phone 14 Brand: Appel Storage: 16GB Camera: 1MP"

    run_pipeline(user_input)
