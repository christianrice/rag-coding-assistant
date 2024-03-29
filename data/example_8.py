"""
tags: [langchain]
description: |
    - Accept string
    - Format single prompt with format instructions using Pydantic
    - Parse response using Pydantic
"""
# Create a chain that does the following:
# - Accept string named query as input
# - Structure the input as an object to pass to the prompt
# - Format the prompt using variables from the object. The prompt includes format instructions using Pydantic
# - Pass the prompt to OpenAI
# - Parse the response using Pydantic
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


parser = PydanticOutputParser(pydantic_object=Actor)

prompt_template = """
Answer the user query.\n
{format_instructions}\n
{query}\n
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI()

chain = {"query": RunnablePassthrough()} | prompt | model | parser

chain.invoke("Generate the filmography for a random actor from Birdman.")
