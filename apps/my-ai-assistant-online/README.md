# My AI Assistant — Online

Online agentic RAG application for querying the knowledge base populated by the offline app.

Install this app from its directory with `pip install -e .`.

## Configuration

Copy `.env.example` to `.env` for local development. The online application
loads typed configuration lazily in `get_agent()`, then injects MongoDB and
OpenAI values into the retriever, summarizer, and agent model. Lower-level
modules do not read environment variables directly.

Use runtime environment variables or a secret manager in deployed
environments. Never commit the local `.env` file.
