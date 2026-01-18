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

def generate_content(topic: str) -> str:
    """Generate initial content on a topic."""
    prompt = f"Write a short paragraph about {topic}. Keep it informative and accurate."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating content: {e}"

def reflect_on_content(content: str) -> str:
    """Reflect and critique the generated content."""
    prompt = f"Review this content ONLY for factual accuracy. If accurate, say 'ACCURATE'. If inaccurate, list only the factual errors:\n\n{content}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error reflecting on content: {e}"

def improve_content(original_content: str, feedback: str) -> str:
    """Improve content based on feedback."""
    prompt = f"Fix ONLY the factual errors mentioned in the feedback. Keep everything else the same:\n\nOriginal: {original_content}\n\nFeedback: {feedback}\n\nProvide corrected version:"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error improving content: {e}"

def reflection_loop(topic: str, iterations: int = 2):
    """Run the reflection loop: Generate → Reflect → Improve."""
    if not llm_available:
        print("LLM not available")
        return
    
    print(f"=== Reflection Loop for: {topic} ===\n")
    
    # Initial generation
    content = generate_content(topic)
    print("INITIAL CONTENT:")
    print(content)
    
    for i in range(iterations):
        print(f"\n--- ITERATION {i+1} ---")
        
        # Reflect
        feedback = reflect_on_content(content)
        print("FEEDBACK:")
        print(feedback)
        
        # Improve
        content = improve_content(content, feedback)
        print("IMPROVED CONTENT:")
        print(content)
    
    print("\n=== FINAL RESULT ===")
    print(content)

def main():
    if not llm_available:
        print("Skipping execution due to LLM initialization failure.")
        return
    
    print("=== AI Reflection System ===")
    print("Enter topics for content generation and improvement (type 'quit' to exit):\n")
    
    while True:
        topic = input("Enter topic: ").strip()
        
        if topic.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not topic:
            print("Please enter a topic.\n")
            continue
            
        reflection_loop(topic)
        print("\n")

if __name__ == "__main__":
    main()