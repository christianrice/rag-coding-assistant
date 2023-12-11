"""
tags: [langchain]
description: |
    - Accept nothing
    - Format System and Human prompts
    - Parse response as string
    - Stream response
"""
# Create a chain that does the following and streams the response:
# - Accept nothing as input
# - Format messages from System and Human as a prompt
# - Pass messages to OpenAI
# - Parse the OpenAI response as a string
# - Stream the response

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser

# Generate system and human messages
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?"),
]

prompt = ChatPromptTemplate.from_messages(messages)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# Stream the chain
for chunk in chain.stream({}):
    print(chunk, end="", flush=True)