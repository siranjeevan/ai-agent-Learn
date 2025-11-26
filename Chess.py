import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env (must contain OPENAI_API_KEY)
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=50,
    api_key=os.getenv("OPENAI_API_KEY"),
)

def interactive_chat(board_position: str):
    """
    Takes a board position as a string and asks the LLM
    for the best move for BLACK. Prints ONLY the move.
    """
    prompt = f"""
You are a strong chess engine playing as BLACK.

Given the following chess board position, written as an 8x8 grid with pieces:

{board_position}

Legend:
- 'B' prefix: black pieces (BR rook, BN knight, BB bishop, BQ queen, BK king, BP pawn)
- 'W' prefix: white pieces (WR, WN, WB, WQ, WK, WP)
- '..' means an empty square.
- Rank 8 is at the top and rank 1 is at the bottom.
- File 'a' is the leftmost column, 'h' is the rightmost column.

Task:
1. Determine the single best legal move for BLACK.
2. Output ONLY the move in standard algebraic notation.
   Examples: e5, Nf6, Qh4+, O-O, exd4
3. Do NOT add any explanation, comments, or extra text.
Just output the move text only.
"""

    try:
        # ✅ Use the correct variable name: prompt
        response = llm.invoke(prompt)
        # Print only the move suggested by the model
        print(response.content.strip())
    except KeyboardInterrupt:
        print("\n👋 Chat ended!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Example board position (your original one)
    user_input = """
    a  b  c  d  e  f  g  h
    8 BR BN BB BQ BK BB .. BR
    7 BP BP BP BP BP BP BP BP
    6 .. .. .. .. .. BN .. ..
    5 .. .. .. .. .. .. .. ..
    4 .. .. .. .. .. .. .. ..
    3 .. .. .. .. .. .. WP WP
    2 WP WP WP WP WP WP .. ..
    1 WR WN WB WQ WK WB WN WR
"""
    interactive_chat(user_input)
