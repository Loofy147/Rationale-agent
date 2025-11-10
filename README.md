# Methodology Automation System (MAS)

This project is an AI-powered system designed to automate and facilitate the "Professional Working Methodology." It acts as an intelligent project manager and engineering assistant, guiding a software project through the six phases of the methodology: Discover, Plan, Implement, Verify, Operate, and Improve.

## Overview

The MAS is built with Python and FastAPI, and it leverages powerful language models from the Hugging Face Hub to automate key tasks in the software development lifecycle.

-   **Discover:** Automatically generates a "Discovery Brief" based on a project topic.
-   **Plan:** Automatically generates a structured "Adaptive Task Plan" from a discovery brief.
-   **(Future) Implement:** Assists with code generation and boilerplate setup.
-   **(Future) Verify:** Orchestrates testing and code analysis.

## Getting Started

1.  **Install dependencies:**
    ```bash
    poetry install
    ```
2.  **Run the application:**
    ```bash
    uvicorn mas.main:app --reload
    ```

## Project Structure

-   `/mas`: The main application source code.
-   `/tests`: Unit and integration tests.
-   `/docs`: Project planning and architecture documents.
-   `openapi.yaml`: The API specification.
-   `pyproject.toml`: Project dependencies and configuration.
