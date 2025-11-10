import uuid
from typing import Dict

class Project:
    def __init__(self, name: str, initial_goal: str):
        self.id = str(uuid.uuid4())
        self.name = name
        self.initial_goal = initial_goal
        self.status = "created"
        self.artifacts = []

class ProjectStateManager:
    """
    Manages the state of all projects in memory.
    (For MVP, this is a simple dictionary. In production, this would be a database.)
    """
    _projects: Dict[str, Project] = {}

    def create_project(self, name: str, initial_goal: str) -> Project:
        project = Project(name, initial_goal)
        self._projects[project.id] = project
        return project

    def get_project(self, project_id: str) -> Project | None:
        return self._projects.get(project_id)

    def update_project_status(self, project_id: str, status: str) -> Project | None:
        project = self.get_project(project_id)
        if project:
            project.status = status
        return project

    def add_artifact(self, project_id: str, artifact_type: str, content: str) -> Project | None:
        project = self.get_project(project_id)
        if project:
            project.artifacts.append({"type": artifact_type, "content": content})
        return project

# Singleton instance of the state manager
state_manager = ProjectStateManager()
