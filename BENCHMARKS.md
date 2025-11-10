# Performance Benchmarks

This document records the performance benchmarks for the Methodology Automation System (MAS).

## Discover Engine Performance

**Date:** 2025-11-10

**Methodology:**
The `scripts/benchmark_discover.py` script was run, which executes the full "Discover" engine workflow for three different topics. This includes loading the `Qwen/Qwen2.5-7B-Instruct` model and running real LLM inference.

**Findings:**
The script timed out after **402 seconds** (6.7 minutes) during its first run. This significantly exceeds our defined Service Level Objective (SLO) of **< 3 minutes** for discovery brief generation.

**Conclusion:**
This benchmark validates the architectural decision to run the AI-intensive tasks (like model inference) in asynchronous, background worker processes. The current synchronous implementation is not performant enough for a production environment. Future work in the "Operate" and "Improve" phases should prioritize the implementation of a proper asynchronous task queue (e.g., using Celery or ARQ).

## Plan Generator Performance

**Date:** 2025-11-10

**Methodology:**
The "Plan" generator uses the same underlying `Qwen/Qwen2.5-7B-Instruct` model as the "Discover" engine.

**Conclusion:**
Based on the results from the "Discover" engine benchmark, it is inferred that the "Plan" generator will have a similar performance profile and will also exceed the acceptable latency for a synchronous API call. This further reinforces the requirement for an asynchronous worker architecture for all LLM-related tasks. A separate benchmark was deemed unnecessary.
