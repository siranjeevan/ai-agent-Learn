# Pipeline Basic (Practice) - Documentation

## 📄 File: `pipeline.py`

### 🔍 Overview
This is a basic practice script that sets up a simple processing pipeline using the Gemini API. It takes user input, sends it to the model, and prints the result. It serves as a foundational building block for more complex chaining.

### 🌊 Workflow Diagram

```mermaid
graph LR
    Input([User Input]) -->|Sends| Scripts[Pipeline Script]
    Scripts -->|API Call| Gemini[Gemini 2.5 Flash]
    Gemini -->|Response| script
    script -->|Print| Output([Console Output])
    
    style Gemini fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Configuration**: setups `genai.configure` with the API key.
2.  **Generation**: Uses `model.generate_content(input)`.
3.  **Loop**: Typically runs in a `while` loop for continuous interaction.

### 🚀 Usage
```bash
venv/bin/python "Prompt Chaining/pipeline.py"
```
