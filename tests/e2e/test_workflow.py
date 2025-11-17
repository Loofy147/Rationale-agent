import pytest
import httpx
import time
import os
# Mark all tests in this file as E2E tests
pytestmark = pytest.mark.e2e

@pytest.fixture(scope="module")
def api_client():
    """
    Provides an httpx client for the E2E tests.
    Assumes the application is running at http://127.0.0.1:8000.
    """
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    with httpx.Client(base_url=base_url) as client:
        yield client

def test_full_workflow(api_client):
    """
    Tests the full user workflow from project creation to plan generation.
    This test relies on the server being run in "TEST_MODE".
    """
    # 1. Create a new project
    response = api_client.post("/projects", json={
        "name": "E2E Test Project",
        "initial_goal": "A test project for E2E testing."
    })
    assert response.status_code == 201
    project = response.json()
    project_id = project["id"]
    assert project["name"] == "E2E Test Project"

    # 2. Initiate the "Discover" phase
    response = api_client.post(f"/projects/{project_id}/discover", json={"topic": "e2e-test"})
    assert response.status_code == 202

    # In a real async setup, we'd poll here. For the MVP, it runs synchronously.
    # Let's just re-fetch the project to check the status.
    response = api_client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    project = response.json()
    assert project["status"] == "discovered"
    assert len(project["artifacts"]) == 1
    assert project["artifacts"][0]["type"] == "discovery_brief"

    # 3. Initiate the "Plan" phase
    response = api_client.post(f"/projects/{project_id}/plan")
    assert response.status_code == 202

    # Re-fetch the project to check the status
    response = api_client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    project = response.json()
    assert project["status"] == "planned"
    assert len(project["artifacts"]) == 2
    assert project["artifacts"][1]["type"] == "adaptive_plan"

    print("E2E test passed successfully.")
