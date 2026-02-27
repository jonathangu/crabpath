# ROADMAP

## Open Work

### Near-term
- Implement chunked/binary storage for graphs with >10K nodes (currently persisted as a single JSON payload).
- Add streaming/incremental traversal via `traverse_stream` (generator-based API for large graphs).
- Enforce CI synchronization checks to ensure `SKILL.md` stays consistent between GitHub and ClawHub.

### Platform extensions
- Add a custom `VectorIndex` backend callback path (HNSW/FAISS-compatible providers).
- Add multi-workspace federation support for querying across multiple brains.
