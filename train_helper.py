import os
from random import random
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import datetime as dt

from tensorflow_core.python.ops.summary_ops_v2 import ResourceSummaryWriter
import tensorflow as tf

from env.env_state import EnvState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.server import Server
from env.task import Task
from eval_results import EvalResults
from resource_weighting_agent import ResourceWeightingAgent
from rl_agents import TaskPricingRLAgent, ResourceWeightingRLAgent, TaskPricingState, ResourceAllocationState
from task_pricing_agent import TaskPricingAgent


def setup_tensorboard(folder: str, training_name: str) -> Tuple[ResourceSummaryWriter, str]:
    """
    为训练和评估结果设置张量板

    Args:
        folder: The folder for the tensorboard
        training_name: Name of the training script
    """
    datetime = dt.datetime.now().strftime("%m-%d_%H-%M-%S")
    print('tensorboard输出的文件夹：', f'{folder}/{training_name}_{datetime}')
    return tf.summary.create_file_writer(f'{folder}/{training_name}_{datetime}'), datetime


def generate_eval_envs(eval_env: OnlineFlexibleResourceAllocationEnv, num_evals: int, folder: str,
                       overwrite: bool = False) -> List[str]:
    """
    生成并保存用于评估智能体训练的评估环境

    Args:
        eval_env: 用来生成文件的评估环境
        num_evals: 生成环境数量
        folder: 生成环境的文件夹
        overwrite: 如果要覆盖以前保存的环境

    Returns: 环境文件路径列表
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    eval_files = []
    for eval_num in range(num_evals):
        eval_file = f'{folder}/eval_{eval_num}.env'
        eval_files.append(eval_file)
        if overwrite or not os.path.exists(eval_file):
            eval_env.reset()
            eval_env.save_env(eval_file)

    return eval_files


def run_training(training_env: OnlineFlexibleResourceAllocationEnv, eval_envs: List[str], total_episodes: int,
                 task_pricing_agents: List[TaskPricingRLAgent],
                 resource_weighting_agents: List[ResourceWeightingRLAgent], eval_frequency: int):
    """
    Runs the training of the agents for a fixed number of episodes

    Args:
        training_env: The training environments
        eval_envs: The evaluation environment filenames
        total_episodes: The total number of episodes
        task_pricing_agents: List of training task pricing agents
        resource_weighting_agents: List of training resource weighting agents
        eval_frequency: The agent evaluation frequency
    """
    # Loop over the episodes
    for episode in range(total_episodes):
        if episode % 5 == 0:
            print(f'Episode: {episode} at {dt.datetime.now().strftime("%H:%M:%S")}')
        train_agent(training_env, task_pricing_agents, resource_weighting_agents)

        # Every eval_frequency episodes, the agents are evaluated
        if episode % eval_frequency == 0:
            eval_agent(eval_envs, episode, task_pricing_agents, resource_weighting_agents)


def train_agent(training_env: OnlineFlexibleResourceAllocationEnv, pricing_agents: List[TaskPricingRLAgent],
                weighting_agents: List[ResourceWeightingRLAgent]):
    """
    Trains reinforcement learning agents through the provided environment

    Args:
        training_env: Training environment used
        pricing_agents: A list of reinforcement learning task pricing agents
        weighting_agents: A list of reinforcement learning resource weighting agents
    """
    # Reset the environment getting a new training environment for this episode
    state = training_env.reset()

    # Allocate the servers with their random task pricing and resource weighting agents
    server_pricing_agents: Dict[Server, TaskPricingRLAgent] = {
        server: random.choice(pricing_agents) for server in state.server_tasks.keys()
    }
    server_weighting_agents: Dict[Server, ResourceWeightingRLAgent] = {
        server: random.choice(weighting_agents) for server in state.server_tasks.keys()
    }

    # Store each server's auction observations with it being (None for first auction because no observation was seen
    # previously) the agent state for the auction (auction task, server tasks, server, time), the action taken and if
    # the auction task was won
    server_auction_states: Dict[Server, Optional[Tuple[TaskPricingState, float, bool]]] = {
        server: None for server in state.server_tasks.keys()
    }

    # For successful auctions, then the agent state of the winning bid, the action taken and the following observation are
    #   all stored in order to be added as an agent observation after the task finishes in order to know if the task was completed or not
    successful_auction_states: List[Tuple[TaskPricingState, float, TaskPricingState]] = []

    # The environment is looped over till the environment is done (the current time step > environment total time steps)
    done = False
    while not done:
        # If the state has a task to be auctioned then find the pricing of each server as the action
        if state.auction_task:
            # Get the bids for each server
            auction_prices = {
                server: server_pricing_agents[server].bid(state.auction_task, tasks, server, state.time_step,
                                                          training=True)
                for server, tasks in state.server_tasks.items()
            }

            # Environment step using the pricing actions to get the next state, rewards, done and info
            next_state, rewards, done, info = training_env.step(auction_prices)

            # Update the server_auction_observations and auction_trajectories variables with the new next_state info
            for server, tasks in state.server_tasks.items():
                # Generate the current agent's state
                current_state = TaskPricingState(state.auction_task, tasks, server, state.time_step)

                if server_auction_states[server]:  # If a server auction observation exists
                    # Get the last time steps agent state, action and if the server won the auction
                    previous_state, previous_action, is_previous_auction_win = server_auction_states[server]

                    # If the server won the auction in the last time step then add the info to the auction trajectories
                    if is_previous_auction_win:
                        successful_auction_states.append((previous_state, previous_action, current_state))
                    else:
                        # Else add the agent state to the agent's replay buffer as a failed auction bid
                        # Else add the observation as a failure to the task pricing rl_agents
                        server_pricing_agents[server].failed_auction_bid(previous_state, previous_action, current_state)

                # Update the server auction agent states with the current agent state
                server_auction_states[server] = (current_state, auction_prices[server], server in rewards)
        else:  # Else the environment is at resource allocation stage
            # For each server and each server task calculate its relative weighting
            weighting_actions: Dict[Server, Dict[Task, float]] = {
                server: server_weighting_agents[server].weight(tasks, server, state.time_step, training=True)
                for server, tasks in state.server_tasks.items()
            }

            # Environment step using the resource weighting actions to get the next state, rewards, done and info
            next_state, finished_server_tasks, done, info = training_env.step(weighting_actions)

            # For each server, there are may be finished tasks due to the resource allocation
            #    therefore add the task pricing auction agent states with the finished tasks
            for server, finished_tasks in finished_server_tasks.items():
                for finished_task in finished_tasks:
                    # Get the successful auction agent state from the list of successful auction agent states
                    successful_auction = next((auction_agent_state
                                               for auction_agent_state in successful_auction_states
                                               if auction_agent_state[0].auction_task == finished_task), None)
                    if successful_auction is None:
                        print(f'Number of successful auction agent states: {len(successful_auction_states)}')
                        print(
                            f'Number of server tasks: {sum(len(tasks) for tasks in next_state.server_tasks.values())}')
                        print(f'Finished task: {str(finished_task)}\n\n')
                        print(f'State: {str(state)}\n')
                        print(f'Next state: {str(next_state)}')
                        break

                    # Remove the successful auction agent state
                    successful_auction_states.remove(successful_auction)

                    # Unwrap the successful auction agent state tuple
                    auction_state, action, next_auction_state = successful_auction

                    # Add the winning auction bid info to the agent
                    server_pricing_agents[server].winning_auction_bid(auction_state, action, finished_task,
                                                                      next_auction_state)

            # Add the agent states for resource allocation
            for server, tasks in state.server_tasks.items():
                agent_state = ResourceAllocationState(tasks, server, state.time_step)
                next_agent_state = ResourceAllocationState(next_state.server_tasks[server], server,
                                                           next_state.time_step)

                server_weighting_agents[server].resource_allocation_obs(agent_state, weighting_actions[server],
                                                                        next_agent_state, finished_server_tasks[server])
        assert all(task.auction_time <= next_state.time_step <= task.deadline
                   for _, tasks in next_state.server_tasks.items() for task in tasks)
        # Update the state with the next state
        state = next_state
def allocate_agents(state: EnvState, task_pricing_agents: List[TaskPricingAgent],
                    resource_weighting_agents: List[ResourceWeightingAgent]) \
        -> Tuple[Dict[Server, TaskPricingAgent], Dict[Server, ResourceWeightingAgent]]:
    """
    Allocates agents to servers

    Args:
        state: Environment state with a list of servers
        task_pricing_agents: List of task pricing agents
        resource_weighting_agents: List of resource weighting agents

    Returns: A tuple of dictionaries, one for the server, task pricing agents and
        the other, server, resource weighting agents
    """
    server_task_pricing_agents = {
        server: random.choice(task_pricing_agents) for server in state.server_tasks.keys()
    }
    server_resource_weighting_agents = {
        server: random.choice(resource_weighting_agents) for server in state.server_tasks.keys()
    }

    return server_task_pricing_agents, server_resource_weighting_agents

def run_training(training_env: OnlineFlexibleResourceAllocationEnv, eval_envs: List[str], total_episodes: int,
                 task_pricing_agents: List[TaskPricingRLAgent],
                 resource_weighting_agents: List[ResourceWeightingRLAgent], eval_frequency: int):
    """
    Runs the training of the agents for a fixed number of episodes

    Args:
        training_env: The training environments
        eval_envs: The evaluation environment filenames
        total_episodes: The total number of episodes
        task_pricing_agents: List of training task pricing agents
        resource_weighting_agents: List of training resource weighting agents
        eval_frequency: The agent evaluation frequency
    """
    # Loop over the episodes
    for episode in range(total_episodes):
        if episode % 5 == 0:
            print(f'Episode: {episode} at {dt.datetime.now().strftime("%H:%M:%S")}')
        train_agent(training_env, task_pricing_agents, resource_weighting_agents)

        # Every eval_frequency episodes, the agents are evaluated
        if episode % eval_frequency == 0:
            eval_agent(eval_envs, episode, task_pricing_agents, resource_weighting_agents)



def eval_agent(env_filenames: List[str], episode: int, pricing_agents: List[TaskPricingAgent],
               weighting_agents: List[ResourceWeightingAgent]) -> EvalResults:
    """
    Evaluation of agents using a list of preset environments

    Args:
        env_filenames: Evaluation environment filenames
        episode: The episode of evaluation
        pricing_agents: List of task pricing agents
        weighting_agents: List of resource weighting agents

    Returns: The evaluation results
    """
    results = EvalResults()

    for env_filename in env_filenames:
        eval_env, state = OnlineFlexibleResourceAllocationEnv.load_env(env_filename)
        server_pricing_agents, server_weighting_agents = allocate_agents(state, pricing_agents, weighting_agents)

        done = False
        while not done:
            if state.auction_task:
                bidding_actions = {
                    server: server_pricing_agents[server].bid(state.auction_task, tasks, server, state.time_step)
                    for server, tasks in state.server_tasks.items()
                }
                state, rewards, done, info = eval_env.step(bidding_actions)
                results.auction(bidding_actions, rewards)
            else:
                weighting_actions = {
                    server: server_weighting_agents[server].weight(tasks, server, state.time_step)
                    for server, tasks in state.server_tasks.items()
                }
                state, rewards, done, info = eval_env.step(weighting_actions)
                results.resource_allocation(weighting_actions, rewards)

        results.finished_env()

    results.save(episode)
    return results
