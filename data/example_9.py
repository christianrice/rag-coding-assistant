"""
tags: [langchain]
description: |
    - Accept string
    - Retrieve matching documents from Weaviate vectorstore
    - Format System and Human messages
    - Parse response as string
    - Stream the response
"""
# Create a chain that does the following:
# - Accept a string as input
# - Retrieve matching documents from a Weaviate vectorstore, and pass through the results and the original request to a prompt as a variable named request
# - Format messages from System and Human using variables from the object to send to OpenAI
# - Pass the prompt to OpenAI
# - Parse the OpenAI response as a string
# - Stream the response
from weaviate import Client
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

client = Client("http://localhost:8080")

def query_collection(query):
    response = (
        client.query
        .get("code_example", ["code"])
        .with_near_text({
            "concepts": [query]
        })
        .with_limit(1)
        .with_additional(["distance"])
        .do()
    )
    return response

system_template = """
Based on this context:\n{context}
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = """
Fulfill this request:\n{request}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

setup_and_retrieval = RunnableParallel(
    {"context": RunnableLambda(query_collection), "request": RunnablePassthrough()}
)

prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = setup_and_retrieval | prompt | model | output_parser

chain.invoke("Here is a request.")