from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, NamedTuple, Dict
import random as rnd
import tensorflow as tf

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage
from resource_weighting_agent import ResourceWeightingAgent
from task_pricing_agent import TaskPricingAgent


class ReinforcementLearningAgent(ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, batch_size: int = 32, error_loss_fn=tf.compat.v1.losses.huber_loss,
                 initial_training_replay_size: int = 5000, training_freq: int = 2, discount_factor: float = 0.9,
                 replay_buffer_length: int = 40000, save_frequency: int = 25000, save_folder: str = 'checkpoint',
                 training_loss_log_freq: int = 250, reward_scaling: float = 1, **kwargs):
        """
        Constructor that is generalised for the deep q networks and policy gradient agents
        Args:
            batch_size: Training batch sizes
            error_loss_fn: Training error loss function
            initial_training_replay_size: The required initial training replay size
            update_frequency: Network update frequency
            discount_factor: TD target discount factor
            replay_buffer_length: Replay buffer length
            save_frequency: Agent save frequency
            save_folder: Agent save folder
            **kwargs:
        """
        assert 0 < batch_size
        assert 0 < training_freq and 0 < save_frequency and 0 < training_loss_log_freq
        assert 0 < initial_training_replay_size and 0 < replay_buffer_length

        # Training
        self.batch_size = batch_size
        self.error_loss_fn = error_loss_fn
        self.initial_training_replay_size = initial_training_replay_size
        self.training_freq = training_freq
        self.training_loss_log_freq = training_loss_log_freq
        self.reward_scaling = reward_scaling
        self.discount_factor = discount_factor

        # Records the agent actions and updates
        self.total_updates: int = 0
        self.total_actions: int = 0

        # Replay buffer
        self.replay_buffer_length = replay_buffer_length
        self.replay_buffer = deque(maxlen=replay_buffer_length)
        self.total_observations: int = 0

        # Save
        self.save_frequency = save_frequency
        self.save_folder = save_folder

    @staticmethod
    def _normalise_task(task: Task, server: Server, time_step: int) -> List[float]:
        """
        Normalises the task that is running on Server at environment time step

        Args:
            task: The task to be normalised
            server: The server that is the task is running on
            time_step: The current environment time step

        Returns: A list of floats where the task attributes are normalised

        """
        return [
            task.required_storage / server.storage_cap,
            task.required_storage / server.bandwidth_cap,
            task.required_computation / server.computational_cap,
            task.required_results_data / server.bandwidth_cap,
            float(task.deadline - time_step),
            task.loading_progress,
            task.compute_progress,
            task.sending_progress
        ]

    def train(self):
        """
        Trains the reinforcement learning agent and logs the training loss
        """
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.replay_buffer, self.batch_size))

        states = tf.keras.preprocessing.sequence.pad_sequences(list(states), dtype='float32')
        actions = tf.cast(tf.stack(actions), tf.float32)  # For DQN, the actions must be converted to int32
        next_states = tf.keras.preprocessing.sequence.pad_sequences(list(next_states), dtype='float32')
        rewards = tf.cast(tf.stack(rewards), tf.float32)
        dones = tf.cast(tf.stack(dones), tf.float32)

        training_loss = self._train(states, actions, next_states, rewards, dones)
        if self.total_updates % self.training_loss_log_freq == 0:
            # noinspection PyUnresolvedReferences
            tf.summary.scalar(f'{self.name} agent training loss', training_loss, self.total_observations)
            tf.summary.scalar(f'Training loss', training_loss, self.total_observations)
        if self.total_updates % self.save_frequency == 0:
            self.save()
        self.total_updates += 1

    @abstractmethod
    def _train(self, states, actions, next_states, rewards, dones) -> float:
        """
        An abstract function to train the reinforcement learning agent

        Args:
            states: Tensor of network observations
            actions: Tensor of actions
            next_states: Tensor of the next network observations
            rewards: Tensor of rewards
            dones: Tensor of dones

        Returns: Training loss
        """
        pass

    @abstractmethod
    def save(self, custom_location: Optional[str] = None):
        """
        Saves a copy of the reinforcement learning agent models at this current total obs
        """
        pass

    def _add_trajectory(self, state, action: float, next_state, reward: float, done: bool = False):
        if done:
            self.replay_buffer.append((state, action, next_state, reward * self.reward_scaling, 0))
        else:
            self.replay_buffer.append((state, action, next_state, reward * self.reward_scaling, 1))

        # Check if to train the agent
        self.total_observations += 1
        if self.initial_training_replay_size <= self.total_observations and \
                self.total_observations % self.training_freq == 0:
            self.train()

    @staticmethod
    def _update_target_network(model_network: tf.keras.Model, target_network: tf.keras.Model, tau: float):
        for model_variable, target_variable in zip(model_network.variables,
                                                   target_network.variables):
            if model_variable.trainable and target_variable.trainable:
                target_variable.assign(tau * model_variable + (1 - tau) * target_variable)


class TaskPricingState(NamedTuple):
    """
    Task pricing reinforcement learning agent state
    """
    auction_task: Task
    tasks: List[Task]
    server: Server
    time_step: int


class TaskPricingRLAgent(TaskPricingAgent, ReinforcementLearningAgent, ABC):
    """
    Task Pricing reinforcement learning agent
    """

    network_obs_width: int = 9

    def __init__(self, name: str, failed_auction_reward: float = -0.05, failed_multiplier: float = -1.5,
                 reward_scaling=0.4, **kwargs):
        """
        Constructor of the task pricing reinforcement learning agent

        Args:
            name: Agent name
            network_input_width: Network input width
            network_output_width: Network output width
            failed_auction_reward: Failed auction reward
            failed_reward_multiplier: Failed reward multiplier
        """
        TaskPricingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, reward_scaling=reward_scaling, **kwargs)

        # Reward variable
        assert failed_auction_reward <= 0, failed_auction_reward
        self.failed_auction_reward = failed_auction_reward
        assert failed_multiplier <= 0, failed_multiplier
        self.failed_multiplier = failed_multiplier

    @staticmethod
    def _network_obs(auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        observation = [ReinforcementLearningAgent._normalise_task(auction_task, server, time_step) + [1.0]] + \
                      [ReinforcementLearningAgent._normalise_task(allocated_task, server, time_step) + [0.0]
                       for allocated_task in allocated_tasks]
        return observation

    def winning_auction_bid(self, agent_state: TaskPricingState, action: float,
                            finished_task: Task, next_agent_state: TaskPricingState):
        """
        When the agent is successful in winning the task then add the task when the task is finished

        Args:
            agent_state: Initial agent state
            action: Auction action
            finished_task: Auctioned finished task containing the winning price
            next_agent_state: Resulting next agent state
        """
        # Check that the arguments are valid
        assert 0 <= action
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED, finished_task

        # Calculate the reward and add it to the replay buffer
        reward = finished_task.price * (1 if finished_task.stage is TaskStage.COMPLETED else self.failed_multiplier)
        obs = self._network_obs(agent_state.auction_task, agent_state.tasks, agent_state.server, agent_state.time_step)
        next_obs = self._network_obs(next_agent_state.auction_task, next_agent_state.tasks,
                                     next_agent_state.server, next_agent_state.time_step)

        self._add_trajectory(obs, action, next_obs, reward)

    def failed_auction_bid(self, agent_state: TaskPricingState, action: float, next_agent_state: TaskPricingState):
        """
        When the agent is unsuccessful in winning the task then add the observation
            and next observation after this action

        Args:
            agent_state: The agent state
            action: The action
            next_agent_state: The next agent state
        """
        # Check that the argument are valid
        assert 0 <= action
        # assert agent_state.time_step <= next_agent_state.time_step

        # If the action is zero then there is no bid on the task so no loss
        obs = self._network_obs(agent_state.auction_task, agent_state.tasks, agent_state.server, agent_state.time_step)
        next_obs = self._network_obs(next_agent_state.auction_task, next_agent_state.tasks,
                                     next_agent_state.server, next_agent_state.time_step)
        self._add_trajectory(obs, action, next_obs, 0 if action == 0 else self.failed_auction_reward)
class ResourceAllocationState(NamedTuple):
    """
    Resource allocation reinforcement learning agent state
    """
    tasks: List[Task]
    server: Server
    time_step: int


class ResourceWeightingRLAgent(ResourceWeightingAgent, ReinforcementLearningAgent, ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    network_obs_width: int = 16

    def __init__(self, name: str, other_task_discount: float = 0.4, success_reward: float = 1,
                 failed_reward: float = -1.5, **kwargs):
        """
        Constructor of the resource weighting reinforcement learning agent

        Args:
            name: The name of the agent
            other_task_discount: The discount for when other tasks are completed
            success_reward: The reward for when tasks have completed successful
            failed_reward: The reward for when tasks have failed
            **kwargs: Additional arguments for the reinforcement learning agent base class
        """
        ResourceWeightingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, **kwargs)

        # Agent reward variables
        assert 0 < other_task_discount
        self.other_task_discount = other_task_discount
        assert failed_reward < 0 < success_reward
        self.success_reward = success_reward
        self.failed_reward = failed_reward

    @staticmethod
    def _network_obs(weighting_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        assert 1 < len(allocated_tasks)

        task_observation = ReinforcementLearningAgent._normalise_task(weighting_task, server, time_step)
        observation = [
            task_observation + ReinforcementLearningAgent._normalise_task(allocated_task, server, time_step)
            for allocated_task in allocated_tasks if weighting_task != allocated_task
        ]

        return observation

    class ResourceAllocationState(NamedTuple):
        """
        Resource allocation reinforcement learning agent state
        """
        tasks: List[Task]
        server: Server
        time_step: int

    def resource_allocation_obs(self, agent_state: ResourceAllocationState, actions: Dict[Task, float],
                                next_agent_state: ResourceAllocationState, finished_tasks: List[Task]):
        """
        Adds a resource allocation state and actions with the resulting resource allocation state with the list of
            finished tasks

        Args:
            agent_state: Resource allocation state
            actions: List of actions
            next_agent_state: Next resource allocation state
            finished_tasks: List of tasks that finished during that round of resource allocation
        """
        # Check that the arguments are valid
        assert len(agent_state.tasks) == len(actions)
        assert all(task in agent_state.tasks for task in actions.keys())
        assert all(0 <= action for action in actions.values())
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)
        assert all(task in next_agent_state.tasks or task in finished_tasks for task in agent_state.tasks)

        if len(agent_state.tasks) <= 1 or len(next_agent_state.tasks) <= 1:
            return

        for task, action in actions.items():
            obs = self._network_obs(task, agent_state.tasks, agent_state.server, agent_state.time_step)
            reward = sum(self.success_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_reward
                         for finished_task in finished_tasks if not task == finished_task) * self.other_task_discount
            if task in next_agent_state.tasks:
                next_task = next(next_task for next_task in next_agent_state.tasks if next_task == task)
                next_obs = self._network_obs(next_task, next_agent_state.tasks, next_agent_state.server,
                                             next_agent_state.time_step)
                self._add_trajectory(obs, action, next_obs, reward)
            else:
                next_obs = np.zeros((1, self.network_obs_width))
                finished_task = next(finished_task for finished_task in finished_tasks if finished_task == task)
                reward += (self.success_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_reward)

                self._add_trajectory(obs, action, next_obs, reward, done=True)


class TaskPricingState(NamedTuple):
    """
    Task pricing reinforcement learning agent state
    """
    auction_task: Task
    tasks: List[Task]
    server: Server
    time_step: int
