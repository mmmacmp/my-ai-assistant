"""Crawl URLs found in extracted documents."""

import asyncio
import logging
from tabnanny import verbose
from uuid import uuid4
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode
from zenml import step


from my_ai_assistant_offline.domain.document import Document, DocumentMetadata
from steps.extract_document import extract_document


logger = logging.getLogger(__name__)


@step
def crawl_links(documents: list[Document], max_concurrent_requests: int = 3,) -> list[Document]:

    if max_concurrent_requests < 1:
        raise ValueError(
            "max_concurrent_requests must be at least 1"
        )

    child_documents = asyncio.run(
        _crawl_links_async(
            documents=documents,
            max_concurrent_requests=max_concurrent_requests,
        )
    )

    augmented_documents = documents + child_documents

    return list(
        {
            document.id: document
            for document in augmented_documents
        }.values()
    )


async def _crawl_links_async(
    documents: list[Document],
    max_concurrent_requests: int,
) -> list[Document]:
    """Crawl document links concurrently."""

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    child_documents: list[Document] = []

    # Create the browser only once and share it between URL tasks.
    async with AsyncWebCrawler(
        verbose=True,
        chrome_channel=None,
    ) as crawler:
        for parent_document in documents:
            tasks = [
                _crawl_url(
                    crawler=crawler,
                    parent_document=parent_document,
                    url=url,
                    semaphore=semaphore,
                )
                for url in parent_document.child_urls
            ]

            # Start all the current parent's URL tasks concurrently.
            results = await asyncio.gather(*tasks)

            child_documents.extend(
                document
                for document in results
                if document is not None
            )

    return child_documents


async def _crawl_url(
    crawler: AsyncWebCrawler,
    parent_document: Document,
    url: str,
    semaphore: asyncio.Semaphore,
) -> Document | None:
    """Crawl one URL while respecting the concurrency limit."""

    async with semaphore:
        logger.info("Crawling: %s", url)

        session_id = uuid4().hex

        try:
            result = await crawler.arun(
                url=url,
                cache_mode=CacheMode.BYPASS,
                session_id=session_id,
            )

            # Small non-blocking delay to avoid hitting websites too quickly.
            await asyncio.sleep(0.5)

        except Exception:
            logger.exception("Failed to crawl %s", url)
            return None

        finally:
            try:
                await crawler.crawler_strategy.kill_session(session_id)
            except Exception:
                logger.exception("Failed to close crawler session for %s", url)

        if not result or not result.success or not result.markdown:
            logger.warning(
                "No usable content found at %s. Error: %s",
                url,
                getattr(result, "error_message", "Unknown error"),
            )
            return None

        properties = dict(result.metadata or {})
        title = properties.pop("title", "") or url

        logger.info(
                "Requested=%s | Result URL=%s | Title=%s | Preview=%r",
                url,
                getattr(result, "url", "Unknown"),
                title,
                str(result.markdown)[:80],
            )

        links = result.links or {}
        child_urls = [
            link["href"]
            for link in (
                links.get("internal", [])
                + links.get("external", [])
            )
            if link.get("href")
        ]

        document_id = uuid4().hex

        return Document(
            id=document_id,
            metadata=DocumentMetadata(
                id=document_id,
                url=url,
                title=title,
                properties=properties,
            ),
            parent_metadata=parent_document.metadata,
            content=str(result.markdown),
            child_urls=list(dict.fromkeys(child_urls)),
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parent = extract_document(
        file_path=Path("data/raw/first_note.md")
    )

    crawled_documents = crawl_links(
        documents=[parent],
        max_concurrent_requests=2,
    )

    print(f"\nSuccessfully crawled: {len(crawled_documents)}")

    for document in crawled_documents:
        print(f"- {document.metadata.title}")
        print(f"  {document.metadata.url}")
