# Understanding RAG

Retrieval-Augmented Generation, usually called RAG, helps an AI answer questions using information from our own documents.

## How it works

A basic RAG system follows these steps:

1. Split documents into smaller chunks.
2. Convert each chunk into an embedding.
3. Store the embeddings in a vector database.
4. Find relevant chunks when the user asks a question.
5. Give those chunks to the language model as context.

## Why use RAG?

RAG helps reduce hallucinations because the model receives relevant information before producing its answer.

## Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [ZenML Documentation](https://docs.zenml.io)