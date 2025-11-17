import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.state_manager import state_manager
from ..engines.discover import discover_engine
from ..engines.plan import plan_generator
from ..data_models import DiscoveryBrief

router = APIRouter()

class ProjectCreate(BaseModel):
    name: str
    initial_goal: str

class DiscoverRequest(BaseModel):
    topic: str

@router.post("", status_code=201)
def create_project(project_create: ProjectCreate):
    """
    Creates a new project.
    """
    project = state_manager.create_project(
        name=project_create.name,
        initial_goal=project_create.initial_goal
    )
    return project

@router.get("/{project_id}")
def get_project(project_id: str):
    """
    Gets project details.
    """
    project = state_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("/{project_id}/discover", status_code=202)
async def run_discover(project_id: str, discover_request: DiscoverRequest):
    """
    Runs the Discover phase for a project.
    """
    project = state_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # For the MVP, we run this synchronously. In production, this would be an async job.
    state_manager.update_project_status(project_id, "discovering")

    # Run the discover engine
    brief = discover_engine.run(topic=discover_request.topic)

    # Add the artifact and update the status
    state_manager.add_artifact(project_id, "discovery_brief", brief.model_dump_json())
    state_manager.update_project_status(project_id, "discovered")

    return {"message": f"Discover phase completed for project {project_id}."}

@router.post("/{project_id}/plan", status_code=202)
async def run_plan(project_id: str):
    """
    Runs the Plan phase for a project.
    """
    project = state_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.artifacts or project.artifacts[-1]["type"] != "discovery_brief":
        raise HTTPException(status_code=400, detail="A discovery brief must be created before the plan phase can be run.")

    # For the MVP, we run this synchronously. In production, this would be an async job.
    state_manager.update_project_status(project_id, "planning")

    # Get the latest discovery brief
    brief_json = project.artifacts[-1]["content"]
    brief = DiscoveryBrief(**json.loads(brief_json))

    # Run the plan generator
    plan = plan_generator.run(brief=brief)

    if plan:
        # Add the artifact and update the status
        state_manager.add_artifact(project_id, "adaptive_plan", plan.model_dump_json())
        state_manager.update_project_status(project_id, "planned")
        return {"message": f"Plan phase completed for project {project_id}."}
    else:
        state_manager.update_project_status(project_id, "planning_failed")
        raise HTTPException(status_code=500, detail="Failed to generate a valid plan.")
