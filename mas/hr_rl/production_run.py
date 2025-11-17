import random
import numpy as np
import torch
import time
import os
from collections import deque

from mas.hr_rl.comprehensive_fix import EnhancedDQNAgent
from mas.hr_rl.environment import PuzzleEnvironment
from mas.hr_rl.core import get_hierarchical_state_representation, get_shaped_reward, decompose_target

def train_agent():
    """Main function to run the training and evaluation with production features."""
    print("Starting Production-Ready Hierarchical RL Training...")

    # --- Configuration ---
    TRAINING_EPISODES = 5000
    STATE_DIM = 12
    ACTION_DIM = 3
    MAX_STEPS_PER_EPISODE = 250
    CHECKPOINT_DIR = "rl_checkpoints"
    CHECKPOINT_FREQ = 500  # Save a checkpoint every 500 episodes
    BEST_MODEL_METRIC = 'avg_reward'
    BEST_MODEL_WINDOW = 100  # Rolling average window for tracking best model

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    agent = EnhancedDQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    # --- Curriculum ---
    curriculum = {
        1: {'range': (5, 10), 'epsilon': 1.0},
        1000: {'range': (12, 18), 'epsilon': 0.7},
        2000: {'range': (20, 30), 'epsilon': 0.5},
        3000: {'range': (5, 100), 'epsilon': 0.3}
    }

    # --- Training Loop ---
    start_time = time.time()
    total_steps = 0
    best_metric_value = -float('inf')
    episode_rewards = deque(maxlen=BEST_MODEL_WINDOW)

    for episode in range(1, TRAINING_EPISODES + 1):
        if episode in curriculum:
            stage_info = curriculum[episode]
            print(f"\nEpisode {episode}: Entering curriculum stage with targets {stage_info['range']}.")
            agent.reset_epsilon(stage_info['epsilon'])

        target_range = next(info['range'] for ep, info in reversed(curriculum.items()) if episode >= ep)
        target = random.randint(*target_range)
        num_forbidden = random.randint(2, 5)
        forbidden_states = set(random.sample(range(1, target), min(num_forbidden, target - 1)))

        env = PuzzleEnvironment(target=target, forbidden_states=forbidden_states, max_steps=MAX_STEPS_PER_EPISODE)

        subgoals = decompose_target(0, target)
        total_reward = 0
        episode_failed = False

        for subgoal in subgoals:
            if episode_failed: break
            env.target = subgoal

            subgoal_reached = False
            while not subgoal_reached:
                current_numeric_state = env.current_state
                env_state = env.get_state()
                state_rep = get_hierarchical_state_representation(
                    current=env_state['current'], target=subgoal,
                    step=env_state['step'], max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'], action_space=env.action_space)

                action_idx = agent.select_action(state_rep)
                action_val = env.action_space[action_idx]
                next_numeric_state, episode_done, info = env.step(action_val)

                next_state_rep = get_hierarchical_state_representation(
                    current=next_numeric_state, target=subgoal,
                    step=env.current_step, max_steps=env.max_steps,
                    forbidden_states=env.forbidden_states, action_space=env.action_space)

                reward = get_shaped_reward(current_numeric_state, next_numeric_state, subgoal, env.current_step, episode_done, info)

                agent.memory.push(state_rep, action_idx, next_state_rep, reward, episode_done)
                agent.train_step()
                total_reward += reward
                total_steps += 1

                if total_steps % 200 == 0:
                    agent.update_target_network()

                if episode_done:
                    episode_failed = True
                    break

                if env.current_state >= subgoal:
                    subgoal_reached = True

        agent.update_epsilon()
        episode_rewards.append(total_reward)

        # --- Logging and Checkpointing ---
        if episode % 100 == 0:
            metrics = agent.get_metrics_summary()
            avg_reward = np.mean(list(episode_rewards))
            print(
                f"E{episode}/{TRAINING_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {metrics.get('avg_loss', 0):.4f} | Q-Val: {metrics.get('avg_q_value', 0):.2f}"
            )

        if episode % CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        current_metric_value = np.mean(list(episode_rewards))
        if len(episode_rewards) == BEST_MODEL_WINDOW and current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(agent.policy_net.state_dict(), best_model_path)
            print(f"New best model saved with avg reward: {best_metric_value:.2f}")


    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.2f} minutes.")

    # Load the best model for evaluation
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for evaluation.")
        agent.policy_net.load_state_dict(torch.load(best_model_path))

    return agent

def evaluate_agent(agent):
    """Evaluates the trained agent on the test cases from the paper."""
    print("\n--- Evaluating Agent ---")

    test_cases = [
        {"target": 123, "forbidden": {23, 45, 67, 89}},
        {"target": 278, "forbidden": {51, 102, 177, 203, 234}},
        {"target": 431, "forbidden": {78, 156, 234, 312, 389}}
    ]

    agent.epsilon = 0.0 # Set to greedy mode for evaluation

    for case in test_cases:
        target = case['target']
        forbidden = case['forbidden']

        env = PuzzleEnvironment(target=target, forbidden_states=forbidden, max_steps=target * 2)
        env.reset()

        path = [0]
        final_state = 0
        evaluation_failed = False

        subgoals = decompose_target(0, target)

        for subgoal in subgoals:
            env.target = subgoal

            subgoal_reached = False
            while not subgoal_reached:
                env_state = env.get_state()
                state_representation = get_hierarchical_state_representation(
                    current=env_state['current'], target=subgoal,
                    step=env_state['step'], max_steps=env_state['max_steps'],
                    forbidden_states=env_state['forbidden_states'], action_space=env.action_space
                )

                action_idx = agent.select_action(state_representation)
                action_val = env.action_space[action_idx]

                next_state, is_terminal, info = env.step(action_val)
                path.append(next_state)
                final_state = next_state

                if is_terminal and info['status'] != 'SUCCESS':
                    evaluation_failed = True
                    break

                if env.current_state >= subgoal:
                    subgoal_reached = True

            if evaluation_failed:
                break

        success = final_state == target
        print(f"\nTarget: {target}, Forbidden: {forbidden}")
        print(f"Result: {'Success' if success else 'Failure'} in {env.current_step} steps.")
        print(f"Path taken: {' -> '.join(map(str, path))}")

if __name__ == "__main__":
    trained_agent = train_agent()
    evaluate_agent(trained_agent)
