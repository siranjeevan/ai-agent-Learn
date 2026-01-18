# Planning Crew (Sequential) - Documentation

## 📄 File: `planning_crew.py`

### 🔍 Overview
This script is a variation of the **Planning Pattern** using CrewAI. Unlike the single-agent planner, this script sets up a "Planner/Writer" agent with a specific task to create a plan *and* a summary in one go, emphasizing the sequential nature of the output.

### 🌊 Workflow Diagram

```mermaid
graph TD
    Topic([Topic Input]) -->|Assign| Agent[Planner Agent]
    
    subgraph "Task Execution"
        Agent -->|Step 1| Plan[Create Outline]
        Plan -->|Step 2| Draft[Write Content]
    end
    
    Draft -->|Result| Output([Final Document])
    
    style Agent fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Agent Config**: `allow_delegation=False` keeps the focus on the single agent's internal process.
2.  **Task Definition**: The `description` explicitly numbers the steps "1. Create a plan... 2. Write the summary...".

### 🚀 Usage
```bash
venv/bin/python "Prompt Chaining/planning_crew.py"
```
