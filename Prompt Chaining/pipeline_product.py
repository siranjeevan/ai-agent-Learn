import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)

prompt_Check_Product = ChatPromptTemplate.from_template(
    "You are a product validator. Check if the product information is factually correct.\n"
    "Product: {text_input}\n\n"
    "Answer with only 'Yes' if all information is correct, or 'No' if there are any errors."
)

prompt_Product_detail = ChatPromptTemplate.from_template(
    "Extract the correct product details for this product in bullet format:\n"
    "Product: {text_input}\n\n"
    "Provide the accurate specifications:"
)

prompt_Message_passing = ChatPromptTemplate.from_template(
    "Compare the provided product data with the correct information:\n\n"
    "Provided Data: {text_input}\n\n"
    "Correct Information: {text_input_2}\n\n"
    "Create a clear comparison table showing any mismatches and ask the user to correct the wrong information."
)

extraction_Product_check = prompt_Check_Product | llm | StrOutputParser()
extraction_Product_detail = prompt_Product_detail | llm | StrOutputParser()
extraction_Message_passing = prompt_Message_passing | llm | StrOutputParser()

full_chain = (
    RunnableMap(
        {
            "text_input": lambda x: x["text_input"],
            "text_input_1": extraction_Product_check,
            "text_input_2": extraction_Product_detail,
        }
    )
    | extraction_Message_passing
)

if __name__ == "__main__":
    input_text = "Product: Samsung Galaxy S24 Brand: Nokia Storage: 16GB Camera: 5MP"

    final_result = full_chain.invoke({"text_input": input_text})

    print(final_result)
