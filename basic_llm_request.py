from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def basic_query(model_name, system_prompt, input, temp=0):
    # check variables
    if model_name is None:
        raise ValueError("model_name is required")
    if system_prompt is None:
        raise ValueError("system_prompt is required")
    if input is None:
        raise ValueError("input is required")

    llm = ChatOpenAI(temperature=temp, model_name=model_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm  | output_parser
    result=chain.invoke({"input": input})
    return result