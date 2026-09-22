# My AI Assistant — Offline

Offline document ETL, summarization dataset generation, and RAG vector-index pipelines.

Install this app from its directory with `pip install -e .`.

## Configuration

Application configuration is loaded and validated by
`my_ai_assistant_offline.config.OfflineSettings`. Copy `.env.example` to `.env`
for local development. Operating-system environment variables override values
from that file.

MongoDB container bootstrap credentials are intentionally separate from the
Python application configuration:

```bash
cp .env.compose.example .env.compose
docker compose --env-file .env.compose up -d mongodb
```

Make the username and password in the application's `MONGODB_URI` match the
local values chosen in `.env.compose`. Neither local file should be committed.
In deployed environments, inject credentials through the runtime or secret
manager instead of shipping dotenv files.

The executable boundary loads configuration once and injects values into the
MongoDB, embedding, retriever, and LLM components. Lower-level modules do not
read environment variables directly.

## Pipelines

Run pipelines as modules from this application directory:

```bash
python -m pipelines.etl
python -m pipelines.dataset_generation
python -m pipelines.rag
```
