# Mnemo — A Cutting-Edge Local-First RAG Engine

![status: wip](https://img.shields.io/badge/status-WIP-orange)

**Working name: Mnemo** (from Mnemosyne, memory). A fully-offline, local-first knowledge engine: point it at your documents, ask questions in natural language, get streamed answers with citations — running entirely on your machine, no API keys. Built to be : usable, demo-able, benchmarkable, and genuinely current with production RAG.

---

## What makes it stand out (the aggressive, cutting-edge bits)

Most "chat with your docs" projects stop at "embed, cosine-search top-k, stuff in prompt." Mnemo does the things that signal _production RAG expertise_:

1. **Contextual Retrieval** (Anthropic's technique) — before embedding each chunk, an LLM prepends a short context describing where the chunk sits in its document. Dramatically improves retrieval; almost no Rust RAG project does this. **This is the headline differentiator.**
2. **Hybrid search + Reciprocal Rank Fusion** — dense (embeddings) _and_ sparse (BM25/full-text) retrieval, fused. Catches both meaning and exact terms (names, codes, IDs).
3. **Cross-encoder reranking** — retrieve wide (50), rerank to the best 8 with a reranker model. The single biggest quality lever.
4. **Agentic / multi-hop retrieval** — query transformation (HyDE, sub-questions) and an agent that can retrieve iteratively for complex questions.
5. **A real evaluation harness** — retrieval metrics (recall@k, MRR, nDCG) + LLM-as-judge for faithfulness/relevance. **This is the rigor signal** — it lets you _prove_ each technique helps, with numbers, in your README.
6. **Streamed answers with inline citations** — every claim traces to a source chunk.
7. **MCP server** — expose your knowledge base as an MCP tool so Claude/any client can query it.
8. **Incremental, watched indexing** — only re-embed changed files.
9. **Fully local & offline** — fastembed (ONNX) + an embedded vector DB + Ollama. No API keys, runs on a laptop.
10. **Observability** — tracing spans, token/latency tracking.

The combination of **contextual retrieval + hybrid + rerank + an eval harness that proves it** is what makes a reviewer think "this person builds production RAG," not "this person did a tutorial."

---

## The stack (local-first, compose-don't-reinvent)

You hand-compose the _pipeline_ (shows judgment) from mature components (reinvent nothing hard). Owning the orchestration — not hiding behind swiftide/rig — reads as more skilled and keeps the repo stable.

| Concern                  | Crate / tool                                                 | Why                                                                            |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Embeddings + reranking   | **`fastembed`**                                              | local ONNX (BGE models), offline, includes rerankers                           |
| Vector + full-text store | **`lancedb`**                                                | embedded (no server), Arrow-based, built-in FTS for hybrid, metadata filtering |
| Chunking                 | **`text-splitter`**                                          | token-aware recursive/semantic splitting                                       |
| LLM generation           | **thin Ollama client** (`reqwest` → OpenAI-compatible `/v1`) | stable, owned, no framework churn                                              |
| CLI + TUI                | **`clap`** + **`ratatui`**                                   | polished interface                                                             |
| MCP server               | **`rmcp`**                                                   | expose to Claude (later milestone)                                             |
| Web API (optional)       | **`axum`**                                                   | demo UI                                                                        |
| Async                    | **`tokio`**                                                  | the runtime                                                                    |
| Errors                   | **`thiserror`** (lib) / **`anyhow`** (bins)                  | typed core, ergonomic edges                                                    |
| Observability            | **`tracing`** + **`tracing-subscriber`**                     | spans, timing                                                                  |
| Testing / bench          | **`proptest`**, **`criterion`**                              | rigor + the eval harness                                                       |

---

## Workspace layout

```
mnemo/
├── Cargo.toml                 # workspace root ([profile.release] strip/lto here)
├── README.md                  # the thing recruiters read — invest here
├── .github/workflows/ci.yml   # fmt + clippy + test + audit
├── crates/
│   ├── mnemo-core/            # the engine: ingest, chunk, embed, index, retrieve
│   │   ├── ingest.rs          # load files (md/txt/pdf), text-splitter chunking
│   │   ├── embed.rs           # fastembed wrapper (embeddings + reranker)
│   │   ├── store.rs           # LanceDB: upsert, vector search, FTS search
│   │   ├── retrieve.rs        # hybrid search + RRF + rerank + query transforms
│   │   ├── context.rs         # Contextual Retrieval (LLM-generated chunk context)
│   │   └── types.rs           # Document, Chunk, Hit, Answer, Citation
│   ├── mnemo-llm/             # thin Ollama client: chat, stream, embed-for-context
│   ├── mnemo-eval/            # eval harness: metrics + LLM-as-judge
│   ├── mnemo-cli/             # clap + ratatui binary (`mnemo`)
│   └── mnemo-mcp/             # MCP server (later)
└── eval/
    └── dataset.jsonl          # labeled Q→relevant-doc pairs for evaluation
```

Layered: `mnemo-core` is pure-ish engine logic; `mnemo-llm` is the generation edge; the binaries are thin adapters. Dependencies flow inward.

---

Email: <anilk.vp@gmail.com>
