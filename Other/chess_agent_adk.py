# Chess Move Suggestion Agent using Google ADK (index-based move output)

import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY must be set in .env file or environment")

GEMINI_MODEL = "gemini-2.0-flash"

# Example board position (text grid, same as before)
BOARD_POSITION = """
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

# --- Chess Engine Agent: INDEX-BASED OUTPUT ---
chess_engine_agent = LlmAgent(
    name="ChessEngine",
    model=GEMINI_MODEL,
    instruction="""
You are a professional chess engine playing as BLACK.

The chess board will be provided as an 8x8 grid with pieces:

Legend:
- 'B' prefix: black pieces (BR rook, BN knight, BB bishop, BQ queen, BK king, BP pawn)
- 'W' prefix: white pieces (WR, WN, WB, WQ, WK, WP)
- '..' means an empty square.
- Rank 8 is at the top and rank 1 is at the bottom.
- File 'a' is the leftmost column, 'h' is the rightmost column.

Internally in code, the board is represented as a 2D JavaScript-style array named "board" where:
- board[0][0] = a8
- board[0][7] = h8
- board[7][0] = a1
- board[7][7] = h1
- Row index increases downward (0 at the top, 7 at the bottom).
- Column index increases left to right (0 = file 'a', 7 = file 'h').

Move format (VERY IMPORTANT):
- You must output the move as: [fromRow, fromCol, toRow, toCol]
- All indexes are zero-based integers.

Your rules:
1. Calculate ONLY legal chess moves for BLACK.
2. Choose the best move for BLACK in the given position.
3. Output ONLY the 4 numbers in array format, like:
   [1,3,3,3]
4. No explanation, no extra text, no chess notation.
5. Do NOT write anything except the numeric array.
""",
    description="Chess engine that suggests the best move for BLACK as index coordinates.",
    output_key="best_move",
)

root_agent = chess_engine_agent

if __name__ == "__main__":
    import asyncio
    import uuid
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    async def get_best_move(board_position: str):
        runner = InMemoryRunner(root_agent, app_name="agents")
        
        user_id = "chess_player"
        session_id = str(uuid.uuid4())
        
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        
        print("🎯 Analyzing chess position...")
        print(f"\nBoard:\n{board_position}")
        
        raw_result = None

        # Simple user message – all rules are in the agent instruction
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "Here is the current chess board position for BLACK:\n"
                            f"{board_position}\n\n"
                            "Return only the best move for BLACK as [fromRow, fromCol, toRow, toCol]."
                        )
                    )
                ],
            ),
        ):
            if event.is_final_response() and event.content:
                if hasattr(event.content, "text") and event.content.text:
                    raw_result = event.content.text
                elif event.content.parts:
                    text_parts = [
                        part.text
                        for part in event.content.parts
                        if hasattr(part, "text") and part.text
                    ]
                    if text_parts:
                        raw_result = "".join(text_parts)

        if not raw_result:
            print("\n⚠️ No result received from the agent")
            return None

        # 🔍 Extract first 4 integers from the response: [fromRow, fromCol, toRow, toCol]
        import re
        nums = re.findall(r"-?\d+", raw_result)
        if len(nums) < 4:
            print("\n⚠️ Could not parse 4 indices from agent response:")
            print("Raw:", raw_result)
            return None

        move = list(map(int, nums[:4]))

        print("\n" + "=" * 40)
        print("♟️  BEST MOVE FOR BLACK (index format)")
        print("=" * 40)
        print(f"  {move}   (format: [fromRow, fromCol, toRow, toCol])")
        print("=" * 40)

        return move

    def main():
        import time

        try:
            asyncio.run(get_best_move(BOARD_POSITION))
            print("\n✅ Chess analysis completed successfully")
        except KeyboardInterrupt:
            print("\n👋 Chess analysis ended!")
        except Exception as e:
            import traceback
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        finally:
            time.sleep(2)

    main()
