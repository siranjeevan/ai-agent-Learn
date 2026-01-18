# Router LLM (Generic) - Documentation

## 📄 File: `router_llm.py`

### 🔍 Overview
This script demonstrates a **Generic LLM Router**. It uses a standard prompting approach to route requests without relying on specific framework features. It's a "Zero-Shot" routing example where the model is asked to output a keyword (e.g., "DATABASE", "API", "UI") which then triggers specific code paths.

### 🌊 Workflow Diagram

```mermaid
graph LR
    Input([Input]) -->|Prompt| LLM[LLM Router]
    LLM -->|Outputs Keyword| Switch{Switch Logic}
    
    Switch -->|'DATABASE'| DB[DB Handler]
    Switch -->|'API'| API[API Handler]
    Switch -->|'UI'| UI[UI Handler]
    
    style LLM fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Router Prompt**: "Classify the following request into one of these categories: [A, B, C]. Return ONLY the category name."
2.  **Control Flow**: Python `if/elif` blocks read the string output and execute the corresponding function.

### 🚀 Usage
```bash
venv/bin/python "Routing/router_llm.py"
```
