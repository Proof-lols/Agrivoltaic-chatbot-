# Agrivoltaics AI Assistant - Evaluation Report

Run: 2026-07-16 22:56 UTC

## Configuration

- Chat model: `gpt-4o`
- Embedding model: `text-embedding-3-small`
- top_k: 5
- Distance threshold: 0.55
- Candidate multiplier: 3
- Chunk size / overlap: 800 / 120
- LLM judge enabled: False

## Summary

- Questions evaluated: 2
- Retrieval hit rate (questions with expected sources): 50% (2 questions)
- Avg groundedness (RAG on): n/a / 5
- Avg groundedness (RAG off): n/a / 5
- Grounded vs ungrounded delta: n/a

## Per-question results

| ID | Hit | Top dist | Grounded | Ungrounded | Question |
|----|-----|----------|----------|------------|----------|
| q01 | no | 0.288 | - | - | What row spacing do I need between agrivoltaic panel rows for mid-size tractors? |
| q02 | yes | 0.462 | - | - | What soil pH range is best for crop rotations under solar panels? |
