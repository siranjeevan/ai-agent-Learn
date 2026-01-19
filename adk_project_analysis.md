
# Analysis of Google ADK Projects

This document provides a technical analysis of the Google Agent Development Kit (ADK) projects found in this repository. The ADK framework uses a declarative style for defining agents (`LlmAgent`, `FunctionTool`) and an execution engine (`InMemoryRunner`) to handle the conversational state and tool loops.

## 1. Reflection Pattern (`Reflection/reflection_adk.py`)

### **Goal**
Implements a "Generator-Critic" loop to improve content quality.

### **Architecture**
*   **Structure:** This file appears to be a *hybrid* implementation. It defines explicit Python functions `generate_draft` and `review_draft` rather than using standard ADK `Agent` classes for orchestration.
*   **Key Logic:**
    1.  **Draft:** Generates initial text.
    2.  **Review:** Calls a separate model prompt to "Fact-check" the text and output JSON logic (`ACCURATE` vs `INACCURATE`).
    3.  **Loop:** The `run_reflection` function manually chains these two calls.
*   **ADK Usage:** Low. It uses `google.generativeai` directly rather than the `google.adk.agents` abstractions found in other files. This serves as a "Manual" reflection pattern.

---

## 2. Routing Pattern (`Routing/adk_router.py` & `Routing/router_machine.py`)

### **Goal**
Routes user queries to specialized agents based on intent.

### **Architecture**
*   **Agents:**
    *   **Coordinator (Parent):** The root agent that receives all messages.
    *   **Sub-Agents:** `Greeter` (LLM-based) and `TaskExecutor` (Code-based).
*   **Routing Logic:**
    *   **Delegation:** The Coordinator's system instruction explicitly tells it *when* to delegate: *"When asked to greet, delegate to Grid..."*.
    *   **Hierarchy:** Uses the `sub_agents=[...]` list to define the valid routing targets.
*   **Special Feature:** `TaskExecutor` implements `BaseAgent` directly (not `LlmAgent`), proving that ADK allows mixing LLMs with deterministic code blocks.
*   **Execution:** `router_machine.py` acts as a complex runner that maintains session history and handles standard routing queries (e.g., "Book me a hotel" vs "Tell me a fact").

---

## 3. Parallelization Pattern (`Parallelization/trip_parallel_adk.py`)

### **Goal**
Generates a comprehensive travel guide by splitting the task into independent domains (Weather, Flights, Hotels, Cost).

### **Architecture**
*   **Agents:** Defined 4 specialized `LlmAgent`s, each with a very narrow system prompt (e.g., "Output *only* the weather...").
*   **Orchestrator:** A `SequentialAgent` named `TravelPlanningPipeline` is used to run them.
    *   *Note:* While the file is named "Parallel", it uses `SequentialAgent` in the code shown. True parallelism in ADK typically uses `ParallelAgent` or `RouterAgent` with non-blocking calls, but this implementation stacks them `[weather, flights, hotels, cost, merger]`.
*   **Synthesis:** The final agent, `TravelSynthesisAgent` (Merger), uses prompt chaining techniques to inject the previous outputs (`{{weather_result}}`) into its context.
*   **Data Flow:** The `output_key` parameter (e.g., `output_key="weather_result"`) is critical; it saves the specific agent's output to the shared context so the Merger agent can read it later.

---

## 4. Other (`Other/chess_agent_adk.py`)

### **Goal**
A specialized tool-use agent designed to play Chess.

### **Architecture**
*   **Agent:** A single `LlmAgent` (`ChessEngine`).
*   **Prompt Engineering:** The prompt is highly constrained to output a specific array format index `[row, col, row, col]`.
*   **Error Handling:** Unlike the conversational agents, this one is designed to be a "backend logic" component where the output is strictly parsed by code (likely regex) rather than shown to a user.

---

## **Summary of ADK Features Demonstrated**

| Feature | Used In | Description |
| :--- | :--- | :--- |
| **LlmAgent** | All except TaskExecutor | Standard agent wrapper for GenAI models. |
| **sub_agents** | `adk_router.py` | Defines hierarchical routing capability. |
| **output_key** | `trip_parallel_adk.py` | Stores intermediate results in the shared execution context for later agents to use. |
| **SequentialAgent** | `trip_parallel_adk.py` | Runs a list of agents in a fixed order. |
| **FunctionTool** | `onboarding_agent_adk.py` | Wraps Python functions so the model can call them (Tool Use). |
| **InMemoryRunner** | All Runners | Manages the chat loop, tool execution, and session state. |

