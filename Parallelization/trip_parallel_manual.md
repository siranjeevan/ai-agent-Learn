# Manual Trip Parallel - Documentation

## 📄 File: `trip_parallel_manual.py`

### 🔍 Overview
This script is a **Manual Implementation** of the Trip Planner parallelization. Instead of using an agent framework, it simply defines Python functions for each domain (Weather, Flight, Hotel) and calls them one by one (or could be async). It shows how to build this pattern from scratch.

### 🌊 Workflow Diagram

```mermaid
graph TD
    Dest([Destination]) -->|Arg| Func1[get_weather()]
    Dest -->|Arg| Func2[get_flights()]
    Dest -->|Arg| Func3[get_hotels()]
    
    Func1 & Func2 & Func3 -->|Strings| Print([Print Results])
    
    style Func1 fill:#e0f2f1,stroke:#00695c,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Functions**: `get_weather_info`, `get_flight_info`, etc., wrap specific prompts.
2.  **Execution**: The `run_travel_agent` function calls these helpers independently. This makes it easy to convert to `asyncio.gather` for true parallelism.

### 🚀 Usage
```bash
venv/bin/python "Parallelization/trip_parallel_manual.py"
```
