"""
tags: [langchain]
description: |
    - Accept object
    - Create multiple chains that work together
    - Parse result as a string
"""
# Create multiple chains that work together to do the following:
# - The first chain should generate a prompt, send it OpenAI, parse the response as a string, and return the response to pass through to the next chain
# - The second chain should generate a prompt, send it OpenAI, parse the response as a string
# - The third chain should generate a prompt, send it OpenAI, parse the response as a string
# - Finally, the fourth chain should format a series of messages using outputs from each of the first chains as AI, Human, and System messages, send those to OpenAI, and parse the result as a string
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


planner = (
    ChatPromptTemplate.from_template("Generate a brief argument about: {input}")
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "List 3 pros or positive aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "List 3 cons or negative aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            (
                "ai",
                "Review your original response (below), and update it based upon the pros and cons.{original_response}",
            ),
            ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),
            ("system", "Generate a final response given the critique"),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

chain.invoke({"input": "scrum"})
