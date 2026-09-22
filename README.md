# My AI Assistant

This repository contains two independently installable applications:

- `apps/my-ai-assistant-offline`: document processing, dataset generation, and RAG indexing.
- `apps/my-ai-assistant-online`: the query-time RAG agent and its tools.
- `apps/infrastructure`: shared local services such as MongoDB.

The online app reads the vector data produced by the offline app, but each app has its own Python package and dependencies.
