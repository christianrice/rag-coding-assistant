"""
tags: [langchain]
description: |
    - Accept string
    - Format single prompt
    - Bind a function
    - Parse response as JSON
"""
# Create a chain that does the following:
# - Accept a string
# - Structure the input as an object to pass to the prompt
# - Format the prompt using variables from the object
# - Bind a function for joke that returns a setup and punchline, and pass the message to OpenAI
# - Parse the response as JSON returning only the setup

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.schema.runnable import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("tell a joke about {foo}")

model = ChatOpenAI(model="gpt-3.5-turbo")

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]

chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

chain.invoke("bears")