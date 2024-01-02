# Overview

RAG Coding Assistant outputs working code in LangChain Expression Language (LCEL) in order to quickly develop new generative AI prototypes.

GPT-4 does not currently have any knowledge of LCEL in its training data, so this tool uses a custom retrieval-augmented generation (RAG) technique to identify **relevant, complete coding examples** and add them to the context window as few-shot examples.

## How It Works

- The `/data` directory is populated with working code examples and some meta content to assist with retrieval
- All code examples from the `/data` directory are embedded in a Weaviate vector database
- When you provide instructions to the assistant for a chain you would like to develop, the assistant retrieves relevant examples from Weaviate and adds them as context to your request
- The assisant will output working code in LCEL according to your specifications

## Current Status

This is a working prototype designed for local development.

## Possible Improvements
1. Improve the embedding and retrieval process: to get high-quality matches, I'm only embedding a shorthand description for each code example that extracts the key specifics of the approach. See an example below where the description that is used in the embedding is a shortened version of the full code comments in the example code:

    ```
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

    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
    from langchain.vectorstores import DocArrayInMemorySearch

    vectorstore = DocArrayInMemorySearch.from_texts(
        ["Lisa likes cooking", "Bears like honey"],
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

    chain.invoke("What does Lisa like?")
    ```

    Once there are enough examples, I might explore creating a fine-tuned model to convert the detailed code comments into that shorthand description in order to both embed the code examples as well as the user request.

2. Expand this beyond LCEL - I included an option in the example code meta tags to specify technologies or keywords. These could be used as hard filters in the Weaviate query, but I'm currently not using them since the results are currently quite accurate. But it could become necessary as the example directory grows.