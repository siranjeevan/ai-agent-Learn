import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env (must contain OPENAI_API_KEY)
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",  # or any other supported model
    temperature=0,
)

# --- Prompt 1: Extract Information ---

def extract_info():
    print("10")

prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text:\n\n{text_input}"
)

prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON object "
    "with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

extraction_chain = prompt_extract | llm | StrOutputParser()

full_chain = ( {"specifications": extraction_chain} | prompt_transform | llm | StrOutputParser() )

if __name__ == "__main__":
    input_text = (
        "The new laptop model features a 3.5 GHz octa-core processor, "
        "16GB of RAM, and a 1TB NVMe SSD."
    )

    # Execute the chain with the input text dictionary.
    final_result = full_chain.invoke({"text_input": input_text})
    

    print("\n--- Final JSON Output ---")
    print(final_result)
    extract_info()
