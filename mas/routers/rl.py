from fastapi import APIRouter, BackgroundTasks
from mas.hr_rl import run as rl_runner

router = APIRouter()

@router.post("/rl/start-training", status_code=202)
async def start_training_and_evaluation(background_tasks: BackgroundTasks):
    """
    Starts the reinforcement learning agent's training and evaluation process
    in the background.
    """

    def run_experiment():
        """A wrapper to run both training and evaluation."""
        trained_agent = rl_runner.train_agent()
        rl_runner.evaluate_agent(trained_agent)

    background_tasks.add_task(run_experiment)

    return {"message": "Reinforcement learning training and evaluation started in the background."}
