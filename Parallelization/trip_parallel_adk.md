# Parallel Trip Planner (ADK) - Documentation

## 📄 File: `trip_parallel_adk.py`

### 🔍 Overview
This script implements a **Parallelization (Sectioning)** pattern. A complex "Plan Trip" request is broken down into four distinct domains (Weather, Flights, Hotels, Cost). Specialized agents generate these sections independently, and a final "Merger" agent synthesizes them into a single report.

### 🌊 Workflow Diagram

```mermaid
graph TD
    Start([Request]) -->|Input| Pipeline
    
    subgraph "Parallel Execution Section"
        Pipeline --> Weather[Weather Agent]
        Pipeline --> Flights[Flights Agent]
        Pipeline --> Hotels[Hotels Agent]
        Pipeline --> Cost[Cost Agent]
    end
    
    Weather -->|Result| Merge{Merger Agent}
    Flights -->|Result| Merge
    Hotels -->|Result| Merge
    Cost -->|Result| Merge
    
    Merge -->|Synthesis| Final([Travel Guide])
    
    style Merge fill:#b2dfdb,stroke:#004d40,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Isolation**: Each sub-agent is instructed to output *only* its specific domain data.
2.  **Merge Prompt**: The `merger_agent` uses a template prompt (e.g., `{{weather_result}}`) to verify that it has received all necessary inputs before generating the final text.
3.  **Sequential vs Parallel**: While currently implemented with `SequentialAgent` for simplicity, the architecture allows these sub-agents to run asynchronously.

### 🚀 Usage
```bash
venv/bin/python "Parallelization/trip_parallel_adk.py"
```
