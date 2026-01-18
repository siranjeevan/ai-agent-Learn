# Reflection Loop - Documentation

## 📄 File: `reflection.py`

### 🔍 Overview
This script checks strict **Reflection (Self-Correction)**. It implements a feedback loop where the model first generates content, then critiques strictly for factual errors, and finally rewrites the content based on that critique.

### 🌊 Workflow Diagram

```mermaid
graph TD
    Start([Topic]) -->|Gen| Draft[Initial Draft]
    
    Draft -->|Review| Critic[Reflection Step]
    Critic -->|Feedback| Check{Errors Found?}
    
    Check -->|Yes| Improve[Improvement Step]
    Improve -->|Refined Text| Draft
    
    Check -->|No| Final([Final Output])
    
    style Critic fill:#ef9a9a,stroke:#c62828,stroke-width:2px
```

### ⚙️ Key Logic
1.  **3-Step Functions**:
    *   `generate_content`: The creative step.
    *   `reflect_on_content`: The analytical step (The Critic).
    *   `improve_content`: The editing step.
2.  **Loop**: The `reflection_loop` function runs these steps for `n` iterations (`iterations=2` by default) to polish the output.

### 🚀 Usage
```bash
venv/bin/python "Reflection/reflection.py"
```
