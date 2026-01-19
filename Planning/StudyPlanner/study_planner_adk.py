
import asyncio
import os
import json
import uuid
import time
from typing import Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

# --- GOOGLE ADK IMPORTS ---
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

# --- 1. MOCK DATA & TOOLS -----------------------------------------------------

# Simulated context for a "Knowledge Base" the agent can query
SUBJECT_METADATA = {
    "Math": {"difficulty": "High", "type": "Problem_Solving"},
    "Physics": {"difficulty": "High", "type": "Problem_Solving"},
    "History": {"difficulty": "Medium", "type": "Memorization"},
    "English": {"difficulty": "Low", "type": "Reading_Writing"},
    "Chemistry": {"difficulty": "High", "type": "Concept_Heavy"},
}

def get_subject_info(subject: str) -> str:
    """
    Retrieves difficulty and type information for a given subject.
    """
    info = SUBJECT_METADATA.get(subject, {"difficulty": "Unknown", "type": "General"})
    return json.dumps(info)

def calculate_priority(exam_date_str: str, subject_difficulty: str) -> str:
    """
    Calculates numerical priority score (1-10) based on urgency and difficulty.
    Date format: YYYY-MM-DD
    """
    try:
        from datetime import datetime
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d")
        days_left = (exam_date - datetime.now()).days
        
        # Difficulty Weight
        diff_weight = 3 if subject_difficulty == "High" else 2 if subject_difficulty == "Medium" else 1
        
        # Urgency Weight
        urgency = 10 if days_left <= 3 else 7 if days_left <= 7 else 4 if days_left <= 14 else 1
        
        score = urgency + diff_weight
        return json.dumps({"subject_priority_score": score, "days_until_exam": days_left})
    except Exception as e:
        return json.dumps({"error": str(e)})

def check_calendar_conflicts(day: str, time_slot: str) -> str:
    """
    Simulates checking a calendar for blocked times.
    """
    # Mocking some busy times
    blocked_times = {
        "Monday": ["09:00-10:00"],
        "Wednesday": ["14:00-15:00"],
    }
    is_blocked = time_slot in blocked_times.get(day, [])
    return json.dumps({"is_available": not is_blocked})

# --- 2. AGENT DEFINITION ------------------------------------------------------

# Wrap tools
tool_subject_info = FunctionTool(get_subject_info)
tool_priority = FunctionTool(calculate_priority)
tool_calendar = FunctionTool(check_calendar_conflicts)

# The Study Planner Agent
study_agent = Agent(
    name="StudyArchitect",
    model="gemini-2.5-flash",
    instruction="""
    You are an intelligent Academic Planning Agent designed using the PLANNING PATTERN.
    Your goal is to create a WEEKLY STUDY SCHEDULE for a student.
    
    You do NOT have a fixed template. You must BUILD the plan by gathering data.
    
    PLANNING STEPS YOU MUST EXECUTE:
    1. ANALYZE INPUTS: Review the student's subjects, exam dates, and preferences provided in the prompt.
    2. DETERMINE PRIORITY: For each subject, use the `calculate_priority` tool to score urgency. 
       Also use `get_subject_info` to understand if it is High/Low difficulty.
    3. CHECK CONSTRAINTS: Use `check_calendar_conflicts` if specific busy times are mentioned (optional).
    4. GENERATE SCHEDULE: Create a 7-day plan. 
       - Give High Priority subjects MORE time slots.
       - Schedule "High Difficulty" subjects during the student's "Preferred Study Time" (Peak Energy).
       - Ensure breaks are included.
       
    OUTPUT FORMAT:
    Return a cleanly formatted markdown response with:
    ### PLAN STRATEGY
    (Bullet points of how you decided priorities)
    
    ### WEEKLY SCHEDULE
    (The timetable)
    
    ### ADAPTABILITY
    (Brief note on what you would change if exam dates moved)
    
    Do not ask the user for more info. Use reasonable assumptions for missing specific dates if needed (e.g., assume exams are 2 weeks away if not stated).
    """,
    tools=[tool_subject_info, tool_priority, tool_calendar],
)

# --- 3. RUNTIME ENGINE --------------------------------------------------------

import logging
logging.basicConfig(level=logging.DEBUG)

async def run_study_planner():
    print("--- 📚 AI Study Planner (Google ADK + Planning Pattern) ---")
    print(f"DEBUG: API Key Loaded? {os.getenv('GOOGLE_API_KEY') is not None}")
    print(f"DEBUG: Content Type: {Content}")
    
    runner = InMemoryRunner(study_agent, app_name="agents")
    
    # STUDENT PROFILE (The Inputs)
    student_profile = """
    Create a study plan for me.
    My Subjects: Math, History, Physics.
    Exam Dates: 
      - Math: 2026-02-01 (Very soon!)
      - Physics: 2026-02-10
      - History: 2026-02-20
    Daily Available Hours: 4 hours (Weekdays), 6 hours (Weekend).
    Weakness: I struggle with Math.
    Strength: I am good at History.
    Preferred Time: I focus best in the Mornings (08:00 - 12:00).
    """
    
    user_id = "student_01"
    session_id = str(uuid.uuid4())
    
    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id, 
        session_id=session_id
    )
    
    print(f"\n📝 STUDENT PROFILE INPUT:\n{student_profile.strip()}\n")
    print("🧠 PLANNING AGENT THINKING...\n")
    
    # Execution Loop with Retry Logic for Rate Limits
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            for event in runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=Content(role="user", parts=[Part(text=student_profile)])
            ):
                # DEBUG PRINT
                print(f"DEBUG LOOP: Event type {type(event)}") 
                
                # Inspect event for logging
                if hasattr(event, 'content'):
                     msg = event.content
                     role = getattr(msg, 'role', 'UNKNOWN').upper()
                     
                     if hasattr(msg, 'parts'):
                         for p in msg.parts:
                             if hasattr(p, 'text') and p.text:
                                 # We print text chunks as they arrive
                                 print(f"[{role}] {p.text}")
                             if hasattr(p, 'function_call') and p.function_call:
                                 print(f"[{role}] 🛠️ REASONING: Checking {p.function_call.name} args={p.function_call.args}")
                             if hasattr(p, 'function_response') and p.function_response:
                                 print(f"[{role}] 📦 DATA RECEIVED: {p.function_response.response}")
            break 

        except Exception as e:
            print(f"⚠️ RAW EXECUTION ERROR: {e}")
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                retry_count += 1
                wait_time = 15 
                print(f"\n⚠️ RATE LIMIT HIT (429). Sleeping {wait_time}s before retry {retry_count}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e 
    
    print("\n✅ PLANNING COMPLETE.")

if __name__ == "__main__":
    asyncio.run(run_study_planner())
