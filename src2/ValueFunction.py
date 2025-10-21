from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment
from Path import Path
from ReplayBuffer import SimpleReplayBuffer, PrioritizedReplayBuffer
from Experience import Experience
from CentralAgent import CentralAgent
from Request import Request
from Experience import Experience

from typing import List, Tuple, Deque, Dict, Any, Iterable

from abc import ABC, abstractmethod
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Masking, Concatenate, Flatten, Bidirectional  # type: ignore
from keras.models import Model, load_model, clone_model  # type: ignore
from keras.backend import function as keras_function  # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras.initializers import Constant  # type: ignore
from tensorflow.summary import FileWriter  # type: ignore
from tensorflow import Summary  # type: ignore
from collections import deque
import numpy as np
from itertools import repeat
from copy import deepcopy
from os.path import isfile, isdir
from os import makedirs
import pickle


class ValueFunction(ABC):
    """docstring for ValueFunction"""

    def __init__(self, log_dir: str):
        super(ValueFunction, self).__init__()

        # Write logs
        log_dir = log_dir + type(self).__name__ + '/'
        if not isdir(log_dir):
            makedirs(log_dir)
        self.writer = FileWriter(log_dir)

    def add_to_logs(self, tag: str, value: float, step: int) -> None:
        summary = Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    @abstractmethod
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, central_agent: CentralAgent):
        raise NotImplementedError

    @abstractmethod
    def remember(self, experience: Experience):
        raise NotImplementedError


class RewardPlusDelay(ValueFunction):
    """docstring for RewardPlusDelay"""

    def __init__(self, DELAY_COEFFICIENT: float=1e-3, log_dir='../logs/'):
        super(RewardPlusDelay, self).__init__(log_dir)
        self.DELAY_COEFFICIENT = DELAY_COEFFICIENT

    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        scored_actions_all_agents: List[List[Tuple[Action, float]]] = []
        for experience in experiences:
            for feasible_actions in experience.feasible_actions_all_agents:
                scored_actions: List[Tuple[Action, float]] = []
                for action in feasible_actions:
                    assert action.new_path

                    immediate_reward = sum([request.value for request in action.requests])
                    remaining_delay_bonus = self.DELAY_COEFFICIENT * action.new_path.total_delay
                    score = immediate_reward + remaining_delay_bonus

                    scored_actions.append((action, score))
                scored_actions_all_agents.append(scored_actions)

        return scored_actions_all_agents

    def update(self, *args, **kwargs):
        pass

    def remember(self, *args, **kwargs):
        pass


class ImmediateReward(RewardPlusDelay):
    """docstring for ImmediateReward"""

    def __init__(self):
        super(ImmediateReward, self).__init__(DELAY_COEFFICIENT=0)


class NeuralNetworkBased(ValueFunction):
    """docstring for NeuralNetwork"""

    def __init__(self, envt: Environment, load_model_loc: str, log_dir: str, GAMMA: float=-1, BATCH_SIZE_FIT: int=32, BATCH_SIZE_PREDICT: int=8192, TARGET_UPDATE_TAU: float=0.1):
        super(NeuralNetworkBased, self).__init__(log_dir)

        # Initialise Constants
        self.envt = envt
        self.GAMMA = GAMMA if GAMMA != -1 else (1 - (0.1 * 60 / self.envt.EPOCH_LENGTH))
        self.BATCH_SIZE_FIT = BATCH_SIZE_FIT
        self.BATCH_SIZE_PREDICT = BATCH_SIZE_PREDICT
        self.TARGET_UPDATE_TAU = TARGET_UPDATE_TAU

        self._epoch_id = 0

        # Get Replay Buffer
        MIN_LEN_REPLAY_BUFFER = 1e6 / self.envt.NUM_AGENTS
        epochs_in_episode = (self.envt.STOP_EPOCH - self.envt.START_EPOCH) / self.envt.EPOCH_LENGTH
        len_replay_buffer = max((MIN_LEN_REPLAY_BUFFER, epochs_in_episode))
        self.replay_buffer = PrioritizedReplayBuffer(MAX_LEN=int(len_replay_buffer))

        # Get NN Model
        self.model: Model = load_model(load_model_loc) if load_model_loc else self._init_NN(self.envt.NUM_LOCATIONS)

        # Define Loss and Compile
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Get target-NN
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Define soft-update function for target_model_update
        self.update_target_model = self._soft_update_function(self.target_model, self.model)

    def _soft_update_function(self, target_model: Model, source_model: Model) -> keras_function:
        target_weights = target_model.trainable_weights
        source_weights = source_model.trainable_weights

        updates = []
        for target_weight, source_weight in zip(target_weights, source_weights):
            updates.append((target_weight, self.TARGET_UPDATE_TAU * source_weight + (1. - self.TARGET_UPDATE_TAU) * target_weight))

        return keras_function([], [], updates=updates)

    @abstractmethod
    def _init_NN(self, num_locs: int):
        raise NotImplementedError()

    @abstractmethod
    def _format_input_batch(self, agents: List[List[LearningAgent]], current_time: float, num_requests: int):
        raise NotImplementedError

    def _get_input_batch_next_state(self, experience: Experience) -> Dict[str, np.ndarray]:
        # Move agents to next states
        all_agents_post_actions = []
        for agent, feasible_actions in zip(experience.agents, experience.feasible_actions_all_agents):
            agents_post_actions = []
            for action in feasible_actions:
                # Moving agent according to feasible action
                agent_next_time = deepcopy(agent)
                assert action.new_path
                agent_next_time.path = deepcopy(action.new_path)
                self.envt.simulate_motion([agent_next_time], rebalance=False)

                agents_post_actions.append(agent_next_time)
            all_agents_post_actions.append(agents_post_actions)

        next_time = experience.time + self.envt.EPOCH_LENGTH

        # Return formatted inputs of these agents
        return self._format_input_batch(all_agents_post_actions, next_time, experience.num_requests)

    def _flatten_NN_input(self, NN_input: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        shape_info: List[int] = []

        for key, value in NN_input.items():
            # Remember the shape information of the inputs
            if not shape_info:
                cumulative_sum = 0
                shape_info.append(cumulative_sum)
                for idx, list_el in enumerate(value):
                    cumulative_sum += len(list_el)
                    shape_info.append(cumulative_sum)

            # Reshape
            NN_input[key] = np.array([element for array in value for element in array])

        return NN_input, shape_info

    def _reconstruct_NN_output(self, NN_output: np.ndarray, shape_info: List[int]) -> List[List[int]]:
        # Flatten output
        NN_output = NN_output.flatten()

        # Reshape
        assert shape_info
        output_as_list = []
        for idx in range(len(shape_info) - 1):
            start_idx = shape_info[idx]
            end_idx = shape_info[idx + 1]
            list_el = NN_output[start_idx:end_idx].tolist()
            output_as_list.append(list_el)

        return output_as_list

    def _format_experiences(self, experiences: List[Experience], is_current: bool) -> Tuple[Dict[str, np.ndarray], List[int]]:
        action_inputs_all_agents = None
        for experience in experiences:
            # If experience hasn't been formatted, format it
            if not (self.__class__.__name__ in experience.representation):
                experience.representation[self.__class__.__name__] = self._get_input_batch_next_state(experience)

            if is_current:
                batch_input = self._format_input_batch([[agent] for agent in experience.agents], experience.time, experience.num_requests)
            else:
                batch_input = deepcopy(experience.representation[self.__class__.__name__])

            if action_inputs_all_agents is None:
                action_inputs_all_agents = batch_input
            else:
                for key, value in batch_input.items():
                    action_inputs_all_agents[key].extend(value)
        assert action_inputs_all_agents is not None

        return self._flatten_NN_input(action_inputs_all_agents)

    def get_value(self, experiences: List[Experience], network: Model=None) -> List[List[Tuple[Action, float]]]:
        # Format experiences
        action_inputs_all_agents, shape_info = self._format_experiences(experiences, is_current=False)

        # Score experiences
        if (network is None):
            expected_future_values_all_agents = self.model.predict(action_inputs_all_agents, batch_size=self.BATCH_SIZE_PREDICT)
        else:
            expected_future_values_all_agents = network.predict(action_inputs_all_agents, batch_size=self.BATCH_SIZE_PREDICT)

        # Format output
        expected_future_values_all_agents = self._reconstruct_NN_output(expected_future_values_all_agents, shape_info)

        # Get Q-values by adding associated rewards
        def get_score(action: Action, value: float):
            return self.envt.get_reward(action) + self.GAMMA * value

        feasible_actions_all_agents = [feasible_actions for experience in experiences for feasible_actions in experience.feasible_actions_all_agents]

        scored_actions_all_agents: List[List[Tuple[Action, float]]] = []
        for expected_future_values, feasible_actions in zip(expected_future_values_all_agents, feasible_actions_all_agents):
            scored_actions = [(action, get_score(action, value)) for action, value in zip(feasible_actions, expected_future_values)]
            scored_actions_all_agents.append(scored_actions)

        return scored_actions_all_agents

    def remember(self, experience: Experience):
        self.replay_buffer.add(experience)

    def update(self, central_agent: CentralAgent, num_samples: int = 3):
        # Check if replay buffer has enough samples for an update
        num_min_train_samples = int(5e5 / self.envt.NUM_AGENTS)
        if (num_min_train_samples > len(self.replay_buffer)):
            return

        # SAMPLE FROM REPLAY BUFFER
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # TODO: Implement Beta Scheduler
            beta = min(1, 0.4 + 0.6 * (self.envt.num_days_trained / 200.0))
            experiences, weights, batch_idxes = self.replay_buffer.sample(num_samples, beta)
        else:
            experiences = self.replay_buffer.sample(num_samples)
            weights = None

        # ITERATIVELY UPDATE POLICY BASED ON SAMPLE
        for experience_idx, (experience, batch_idx) in enumerate(zip(experiences, batch_idxes)):
            # Flatten experiences and associate weight of batch with every flattened experience
            if weights is not None:
                weights = np.array([weights[experience_idx]] * self.envt.NUM_AGENTS)

            # GET TD-TARGET
            # Score experiences
            scored_actions_all_agents = self.get_value([experience], network=self.target_model)  # type: ignore

            # Run ILP on these experiences to get expected value at next time step
            value_next_state = []
            for idx in range(0, len(scored_actions_all_agents), self.envt.NUM_AGENTS):
                final_actions = central_agent.choose_actions(scored_actions_all_agents[idx:idx + self.envt.NUM_AGENTS], is_training=False)
                value_next_state.extend([score for _, score in final_actions])

            supervised_targets = np.array(value_next_state).reshape((-1, 1))

            # UPDATE NN BASED ON TD-TARGET
            action_inputs_all_agents, _ = self._format_experiences([experience], is_current=True)
            history = self.model.fit(action_inputs_all_agents, supervised_targets, batch_size=self.BATCH_SIZE_FIT, sample_weight=weights)

            # Write to logs
            loss = history.history['loss'][-1]
            self.add_to_logs('loss', loss, self._epoch_id)

            # Update weights of replay buffer after update
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                # Calculate new squared errors
                predicted_values = self.model.predict(action_inputs_all_agents, batch_size=self.BATCH_SIZE_PREDICT)
                loss = np.mean((predicted_values - supervised_targets) ** 2 + 1e-6)
                # Update priorities
                self.replay_buffer.update_priorities([batch_idx], [loss])

            # Soft update target_model based on the learned model
            self.update_target_model([])

            self._epoch_id += 1


class PathBasedNN(NeuralNetworkBased):

    def __init__(self, envt: Environment, load_model_loc: str='', log_dir: str='../logs/'):
        super(PathBasedNN, self).__init__(envt, load_model_loc, log_dir)

    def _init_NN(self, num_locs: int) -> Model:
        # DEFINE NETWORK STRUCTURE
        # Check if there are pretrained embeddings
        if (isfile(self.envt.DATA_DIR + 'embedding_weights.pkl')):
            weights = pickle.load(open(self.envt.DATA_DIR + 'embedding_weights.pkl', 'rb'))
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding', embeddings_initializer=Constant(weights[0]), trainable=False)
        else:
            location_embed = Embedding(output_dim=100, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding')

        # Get path and current locations' embeddings
        path_location_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32', name='path_location_input')
        path_location_embed = location_embed(path_location_input)

        # Get associated delay for different path locations
        delay_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1), name='delay_input')
        delay_masked = Masking(mask_value=-1)(delay_input)

        # Get entire path's embedding
        path_input = Concatenate()([path_location_embed, delay_masked])
        path_embed = LSTM(200, go_backwards=True)(path_input)

        # Get current time's embedding
        current_time_input = Input(shape=(1,), name='current_time_input')
        current_time_embed = Dense(100, activation='elu', name='time_embedding')(current_time_input)

        # Get embedding for other agents
        other_agents_input = Input(shape=(1,), name='other_agents_input')

        # Get embedding for number of requests
        num_requests_input = Input(shape=(1,), name='num_requests_input')

        # Get Embedding for the entire thing
        state_embed = Concatenate()([path_embed, current_time_embed, other_agents_input, num_requests_input])
        state_embed = Dense(300, activation='elu', name='state_embed_1')(state_embed)
        state_embed = Dense(300, activation='elu', name='state_embed_2')(state_embed)

        # Get predicted Value Function
        output = Dense(1, name='output')(state_embed)

        model = Model(inputs=[path_location_input, delay_input, current_time_input, other_agents_input, num_requests_input], outputs=output)

        return model

    def _format_input(self, agent: LearningAgent, current_time: float, num_requests: float, num_other_agents: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        # Normalising Inputs
        current_time_input = (current_time - self.envt.START_EPOCH) / (self.envt.STOP_EPOCH - self.envt.START_EPOCH)
        num_requests_input = num_requests / self.envt.NUM_AGENTS
        num_other_agents_input = num_other_agents / self.envt.NUM_AGENTS

        # Getting path based inputs
        location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32')
        delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1)) - 1

        # Adding current location
        location_order[0] = agent.position.next_location + 1
        delay_order[0] = 1

        for idx, node in enumerate(agent.path.request_order):
            if (idx >= 2 * self.envt.MAX_CAPACITY):
                break

            location, deadline = agent.path.get_info(node)
            visit_time = node.expected_visit_time

            location_order[idx + 1] = location + 1
            delay_order[idx + 1, 0] = (deadline - visit_time) / Request.MAX_DROPOFF_DELAY  # normalising

        return location_order, delay_order, current_time_input, num_requests_input, num_other_agents_input

    def _format_input_batch(self, all_agents_post_actions: List[List[LearningAgent]], current_time: float, num_requests: int) -> Dict[str, Any]:
        input: Dict[str, List[Any]] = {"path_location_input": [], "delay_input": [], "current_time_input": [], "other_agents_input": [], "num_requests_input": []}

        # Format all the other inputs
        for agent_post_actions in all_agents_post_actions:
            current_time_input = []
            num_requests_input = []
            path_location_input = []
            delay_input = []
            other_agents_input = []

            # Get number of surrounding agents
            current_agent = agent_post_actions[0]  # Assume first action is _null_ action
            num_other_agents = 0
            for other_agents_post_actions in all_agents_post_actions:
                other_agent = other_agents_post_actions[0]
                if (self.envt.get_travel_time(current_agent.position.next_location, other_agent.position.next_location) < Request.MAX_PICKUP_DELAY or
                        self.envt.get_travel_time(other_agent.position.next_location, current_agent.position.next_location) < Request.MAX_PICKUP_DELAY):
                    num_other_agents += 1

            for agent in agent_post_actions:
                # Get formatted output for the state
                location_order, delay_order, current_time_scaled, num_requests_scaled, num_other_agents_scaled = self._format_input(agent, current_time, num_requests, num_other_agents)

                current_time_input.append(current_time_scaled)
                num_requests_input.append(num_requests)
                path_location_input.append(location_order)
                delay_input.append(delay_order)
                other_agents_input.append(num_other_agents_scaled)

            input["current_time_input"].append(current_time_input)
            input["num_requests_input"].append(num_requests_input)
            input["delay_input"].append(delay_input)
            input["path_location_input"].append(path_location_input)
            input["other_agents_input"].append(other_agents_input)

        return input
