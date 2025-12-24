# Model Fallback Demo - Simplified version
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Model candidates for fallback (JavaScript logic ported to Python)
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro", 
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

def get_working_model(candidates=MODEL_CANDIDATES):
    """Find the first working Gemini model from candidates"""
    print("🔍 Testing model candidates...")
    
    for model_name in candidates:
        try:
            print(f"   Testing {model_name}...")
            
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0,
                }
            )
            
            # Test the model with a simple prompt (equivalent to "ok" in JS)
            response = model.generate_content("Hello")
            
            print(f"✅ Successfully connected to model: {model_name}")
            return model
            
        except Exception as e:
            print(f"❌ Model failed: {model_name} - {str(e)}")
    
    print("⚠️  No working Gemini model found")
    return None

def demonstrate_model_usage():
    """Demonstrate using the working model"""
    print("\n" + "="*60)
    print("🚀 GEMINI MODEL FALLBACK DEMONSTRATION")
    print("="*60)
    
    # Get working model using fallback logic
    working_model = get_working_model()
    
    if working_model:
        print(f"\n🎯 Using working model to generate content...")
        
        # Test prompt
        prompt = """Create a brief plan and summary about the importance of Reinforcement Learning in AI.
        
Format your response as:
### Plan
- [bullet points]

### Summary  
- [brief summary]"""
        
        try:
            response = working_model.generate_content(prompt)
            print("\n📝 Generated Response:")
            print("-" * 40)
            print(response.text)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error generating content: {e}")
    else:
        print("\n🔄 Demonstrating fallback behavior...")
        print("Since no models are available, here's what the fallback would return:")
        print("-" * 40)
        print("""### Plan
- Introduction to Reinforcement Learning (RL)
- Key concepts and terminology  
- Applications in AI systems
- Benefits and advantages
- Current challenges and limitations
- Future prospects

### Summary
Reinforcement Learning represents a crucial paradigm in artificial intelligence that enables systems to learn optimal behaviors through interaction with their environment. Unlike supervised learning, RL agents learn through trial and error, receiving rewards or penalties based on their actions. This approach has proven invaluable in complex decision-making scenarios such as game playing, robotics, autonomous vehicles, and recommendation systems. The importance of RL lies in its ability to handle sequential decision-making problems where consequences may not be immediately apparent. Key advantages include adaptability to changing environments, ability to discover novel strategies, and applicability where labeled training data is scarce. However, RL faces challenges including sample efficiency, exploration vs exploitation trade-offs, and training stability. Despite these challenges, RL continues to drive AI breakthroughs, with recent advances in deep reinforcement learning opening new possibilities for solving previously intractable problems.""")
        print("-" * 40)

if __name__ == "__main__":
    demonstrate_model_usage()