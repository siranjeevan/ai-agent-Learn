# python Planning/Deep Research/gemini_deep_research.py



import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini with API key from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the agent's role and the user's research question
system_message = """You are a professional researcher preparing
a structured, data-driven report.
Focus on data-rich insights, use reliable sources, and include
inline citations."""

user_query = "Research the economic impact of semaglutide on global healthcare systems."

# Create Gemini 2.5 Flash model
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_message
)

# Generate the research report
response = model.generate_content(
    user_query,
    generation_config=genai.GenerationConfig(
        temperature=0,
        max_output_tokens=4096
    )
)

# Print the final report
final_report = response.text
print(final_report)

# --- LIMITATION NOTICE ---
print("\n--- NOTE ---")
print("Gemini API does not expose:")
print("- Inline citation metadata objects")
print("- Internal reasoning steps")
print("- Web search execution logs")
print("- Code interpreter traces")
