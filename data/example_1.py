"""
tags: [langchain]
description: |
    - Accept object
    - Format single prompt
    - Parse response as string
"""
# Create a chain that performs the following steps:
# - Accepts an object as input.
# - Formats the prompt using variables from the object.
# - Sends the prompt to OpenAI.
# - Parses the response as a string.

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})
