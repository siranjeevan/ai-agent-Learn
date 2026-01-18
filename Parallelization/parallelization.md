# Parallelization (Simple Manual) - Documentation

## 📄 File: `parallelization.py`

### 🔍 Overview
This script shows a basic, manual implementation of the **Parallelization** pattern. It simulates a scenario where an "Orchestrator" distributes a complex query to two different sub-models (or the same model with different personas) and creates a final result from their combined outputs.

### 🌊 Workflow Diagram

```mermaid
graph TD
    Input([User Query]) -->|Copy| Orchestrator
    
    Orchestrator -->|Sub-Task A| ModelA[Model A (Data)]
    Orchestrator -->|Sub-Task B| ModelB[Model B (Creative)]
    
    ModelA -->|Result| Combiner
    ModelB -->|Result| Combiner
    
    Combiner -->|Formatting| Final([Final Answer])
```

### ⚙️ Key Logic
1.  **Concurrent Execution**: (In a full production version/async) `ModelA` and `ModelB` would run at the exact same time.
2.  **Persona splitting**: One path might focus on "Extraction" while the other focuses on "Validation".
3.  **Aggregation**: The final step is purely algorithmic or a simple LLM call to merge the two streams.

### 🚀 Usage
```bash
venv/bin/python "Parallelization/parallelization.py"
```
