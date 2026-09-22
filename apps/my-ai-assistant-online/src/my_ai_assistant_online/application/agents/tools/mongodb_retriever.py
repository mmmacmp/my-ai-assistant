from smolagents import Tool

from my_ai_assistant_online.application.rag.retrievers import get_retriever
from my_ai_assistant_online.application.rag.types import (
    EmbeddingModelType,
    RetrieverType,
)


class MongoDBRetrieverTool(Tool):
    name = "mongodb_retriever"
    description = """Use this tool to search and retrieve relevant documents from the knowledge base
    using semantic search. Best used when you need specific information, context about a topic, or
    supporting sources for an answer. Returns multiple relevant document snippets with title, url,
    and content.""".strip()
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query describing information to be retrieved",
        }
    }
    output_type = "string"

    def __init__(
        self,
        collection_name: str,
        embedding_model_id: str,
        embedding_model_type: EmbeddingModelType,
        retriever_type: RetrieverType,
        k: int = 5,
        device: str = "cpu",
        *,
        mongodb_uri: str,
        database_name: str,
        openai_api_key: str | None = None,
    ) -> None:
        super().__init__()

        self.retriever = get_retriever(
            collection_name=collection_name,
            device=device,
            embedding_model_id=embedding_model_id,
            embedding_model_type=embedding_model_type,
            k=k,
            retriever_type=retriever_type,
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            openai_api_key=openai_api_key,
        )

    def forward(self, query: str) -> str:
        documents = self.retriever.invoke(query)
        if not documents:
            return "No relevant documents were found"

        formatted_documents = []

        for index, doc in enumerate(documents, start=1):
            title = doc.metadata.get("title", "Unknown title")
            url = doc.metadata.get("url", "Unknown url")
            content = doc.page_content.strip()

            formatted_document = f"""
            <document id="{index}">
            <title>{title}</title>
            <url>{url}</url>
            <content>
            {content}
            </content>
            </document>
            """.strip()

            formatted_documents.append(formatted_document)

        joined_documents = "\n\n".join(formatted_documents)

        return f"""
        <search_results>
        {joined_documents}
        </search_results>
        When using information from a document, include its url as a reference""".strip()
