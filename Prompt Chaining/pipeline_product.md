# Product Extractor Pipeline - Documentation

## 📄 File: `pipeline_product.py`

### 🔍 Overview
This script implements an **Extraction Chain**. Its purpose is to take unstructured text (e.g., product descriptions, emails) and extract structured product data (Name, Price, SKU, Features) into a clean format (JSON or list).

### 🌊 Workflow Diagram

```mermaid
graph TD
    Raw([Raw Text]) -->|Input| Extractor[Extraction Agent]
    
    subgraph "Extraction Logic"
        Extractor -->|Identify| Name[Product Name]
        Extractor -->|Identify| Specs[Specifications]
        Extractor -->|Identify| Price[Price/Cost]
    end
    
    Name & Specs & Price -->|Format| JSON([Structured Data])
    
    style Extractor fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

### ⚙️ Key Logic
1.  **Prompt Engineering**: The prompt is tuned to "Extract" rather than "Generate".
2.  **Formatting**: Often instructs the model to return JSON or Key-Value pairs for easy programmatic parsing.

### 🚀 Usage
```bash
venv/bin/python "Prompt Chaining/pipeline_product.py"
```
