import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)
prompt_fix_grammar = ChatPromptTemplate.from_template(
    "Please correct the grammar mistakes in the following text and No change the :\n\n{text_input}"
)

prompt_uppercase = ChatPromptTemplate.from_template(
    "Convert the following text into UPPERCASE:\n\n{text_input_2}"
)

prompt_remove_space = ChatPromptTemplate.from_template(
    "Remove all unwanted spaces from the following text:\n\n{text_input_1}"
)

extraction_uppercase = prompt_uppercase | llm | StrOutputParser()
extraction_remove_space = prompt_remove_space | llm | StrOutputParser()
extraction_fix_grammar = prompt_fix_grammar | llm | StrOutputParser()

full_chain = (
    RunnableMap({"text_input_1": extraction_fix_grammar})
    | RunnableMap({"text_input_2": extraction_remove_space})
    | extraction_uppercase
)

if __name__ == "__main__":
    input_text = "hi i am Jeevith                        . how are from  "
    final_result = full_chain.invoke({"text_input": input_text})
    print(final_result)
