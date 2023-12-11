"""
tags: [langchain]
description: |
    - Accept string
    - Format single prompt
    - Parse response as string
"""
# Create a chain that does the following:
# - Accept a string as input
# - Structure the input as an object to pass to the prompt
# - Format the prompt using variables from the object
# - Send the prompt to OpenAI
# - Parse the response as a string

from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
output_parser = StrOutputParser()
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = (
    {"topic": RunnablePassthrough()} 
    | prompt
    | model
    | output_parser
)

chain.invoke("ice cream")