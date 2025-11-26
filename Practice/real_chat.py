import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize with real API key
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

def interactive_chat():
    print("🤖 Real AI Chat - Connected!")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n💬 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            response = llm.invoke(user_input)
            print(f"🤖 AI: {response.content}")
            
        except KeyboardInterrupt:
            print("\n👋 Chat ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat()