# Simple Router (ADK) - Documentation

## 📄 File: `adk_router.py`

### 🔍 Overview
This script shows a simpler hierarchical **Routing/Delegation** example. A `Coordinator` agent serves as the parent and routes tasks between a "Greeter" (LLM-based) and a "TaskExecutor" (Code-based/Deterministic).

### 🌊 Workflow Diagram

```mermaid
graph TD
    User([User]) -->|Chat| Coord[Coordinator Agent]
    
    Coord -->|'Hello'| Greeter[Greeter Agent (LLM)]
    Coord -->|'Do Task'| Executor[TaskExecutor (Code)]
    
    Executor -->|Code Logic| Event[Task Event]
    Greeter -->|LLM Gen| Reply[Text Reply]
    
    style Executor fill:#e0e0e0,stroke:#333,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Hybrid Agents**: Demonstrates mixing `LlmAgent` (Greeter) with a custom class `TaskExecutor(BaseAgent)`. This allows for routing to strict code logic rather than just another prompt.
2.  **Parent-Child Relationship**: The `sub_agents` list defines the hierarchy.

### 🚀 Usage
```bash
venv/bin/python "Routing/adk_router.py"
```
