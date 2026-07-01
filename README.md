# Bashqueen Bot

An async Telegram bot scaffold built with `aiogram` and the xAI API. The project is structured for long-term maintenance: clear service boundaries, SQLite persistence, tool-calling support, task scheduling, local media discovery, and code-level safety-state handling.

This scaffold intentionally keeps the runtime persona in a stern, immersive, command-oriented lane without implementing explicit sexual humiliation or graphic erotic scripting. The architecture still supports:

- Telegram command and message handling
- xAI Responses API integration with custom function calls
- SQLite-backed user profiles, tasks, state, and recent message history
- Safeword interception and automatic aftercare / pause transitions
- Task frequency windows with skip-on-ignore behavior
- Local image / video library discovery and optional image generation
- Docker deployment for NAS-friendly hosting

## Quick Start

1. Copy `.env.example` to `.env` and fill in `BOT_TOKEN` and `XAI_API_KEY`.
2. Put local media into `assets/images` and `assets/videos`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the bot:

```bash
python -m bot.main
```

## Docker

```bash
docker compose up --build -d
```

The SQLite database is persisted under `./data`, and assets are mounted from `./assets`.

## Tests

```bash
pytest
```

