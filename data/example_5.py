"""
tags: [langchain]
description: |
    - Accept object
    - Format System and Human prompts
    - Do not parse response
"""
# Create a chain that does the following:
# - Accept an object
# - Format messages from System and Human using variables from the object
# - Pass messages to OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

model = ChatOpenAI()

chain = chat_prompt | model

chain.invoke({'input_language': 'English', 'output_language': 'French', 'text': 'Hello, how are you?'})