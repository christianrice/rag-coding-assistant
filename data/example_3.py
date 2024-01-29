"""
tags: [langchain, rag]
description: |
    - Accept string
    - Retrieve matching documents using DocArrayInMemorySearch vectorstore
    - Format single prompt
    - Parse response as string
"""
# Create a chain that does the following:
# - Accept a string as input
# - Retrieve matching documents from a DocArrayInMemorySearch vectorstore, and pass through the results and the original question to a prompt
# - Format the prompt using variables from the object
# - Pass the prompt to OpenAI
# - Parse the OpenAI response as a string

from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["Tom likes clouds", "bears like honey"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

template = """Answer the question:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser

chain.invoke("what does Tom like?")
