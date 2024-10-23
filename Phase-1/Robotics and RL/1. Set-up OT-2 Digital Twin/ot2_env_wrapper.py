import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents = 1)

        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Box(-1, 1, (3,), np.float32)

        # Define observation space with only pipette_position
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        # keep track of the number of steps
        self.steps = 0


    def reset(self, seed=None):
            # being able to set a seed is required for reproducibility
            if seed is not None:
                np.random.seed(seed)

            # Reset the state of the environment to an initial state
            # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
            self.goal_position = np.array([0.1439, 0.1603, 0.1195], dtype=np.float32)

            # Call the environment reset function
            observation = self.sim.reset(num_agents=1)

            # Get the first key in the dictionary (robot ID)
            robot_id = next(iter(observation))

            # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
            pipette_position = observation[robot_id]['pipette_position']

            # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
            #pipette_position = observation['robotId_1']['pipette_position']

            # Append the goal position to the pipette position
            observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

            # Reset the number of steps
            self.steps = 0

            inf = {}

            return (observation, inf)
    

    def step(self, action):
            # Execute one time step within the environment
            # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
            action = np.append(action, 0.0)

            # Call the environment step function
            observation = self.sim.run([action]) # Why do we need to pass the action as a list? Because the simulation class expects a list of actions


            # Get the first key in the dictionary (robot ID)
            robot_id = next(iter(observation))

            # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
            pipette_position = observation[robot_id]['pipette_position']

            # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
            #pipette_position = observation['robotId_1']['pipette_position']
            observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

            distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

            reward = - distance_to_goal
            
            # next we need to check if the if the task has been completed and if the episode should be terminated
            # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
            # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.

            if distance_to_goal < 30:
                terminated = True
                # we can also give the agent a positive reward for completing the task
            else:
                terminated = False

            # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
            if self.steps >= self.max_steps:
                truncated = True
            else:
                truncated = False

            info = {} # we don't need to return any additional information

            # increment the number of steps
            self.steps += 1

            return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()