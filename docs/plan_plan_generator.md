# Implementation Plan: Plan Generator

This document breaks down the tasks required to build the `Plan Generator` for the MAS MVP.

## Epic: Implement MAS MVP
### Feature: Implement Plan Generator

| ID | Type | Title | Description | Estimate | Priority | Risk | Dependencies |
|---|---|---|---|---|---|---|---|
| T-7 | Task | Develop Prompt Template for Planning | Create a robust prompt template that instructs the `Qwen/Qwen2.5-7B-Instruct` model to generate a plan in the "Adaptive Task Plan" format. | 4h | P0 | H | - |
| T-8 | Task | Implement Planning Service | Create a service that takes a `DiscoveryBrief` artifact, uses the prompt template to call the LLM, and returns the raw text of the generated plan. | 5h | P0 | M | T-7 |
| T-9 | Task | Create `AdaptiveTaskPlan` Data Models | Define the Pydantic models for the `AdaptiveTaskPlan` artifact, including structures for Epics, Features, and Tasks. | 3h | P1 | L | - |
| T-10 | Task | Implement Plan Parser | Create a parser that can reliably convert the LLM's markdown output into the structured `AdaptiveTaskPlan` data models. This may involve using regex or another LLM call for extraction. | 6h | P0 | H | T-9 |
| T-11 | Task | Orchestrate Plan Generation Workflow | Create the main `PlanGenerator` class that uses the other services to perform the end-to-end plan generation process: prompt, generate, parse, and validate. | 4h | P0 | M | T-8, T-10 |
| T-12 | Task | Add Unit Tests for Parser and Models | Write unit tests for the `AdaptiveTaskPlan` data models and the `PlanParser`. | 4h | P1 | L | T-9, T-10 |
| T-13 | Task | Add Integration Test for Plan Generator | Write an integration test that runs the full plan generation workflow, using a real `DiscoveryBrief` and mocking the LLM call. | 5h | P1 | M | T-11 |
