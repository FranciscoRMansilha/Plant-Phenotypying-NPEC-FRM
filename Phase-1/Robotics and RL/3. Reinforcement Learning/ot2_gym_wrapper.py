import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.render_mode = render

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        self.previous_position = np.asarray([0.095, 0.1153, 0.1295]
        )


        # Define action and observation space They must be gym.spaces objects
        self.action_space = spaces.Box(-1, 1, (3,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (6,), np.float32)

        working_envelope = {'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}

        # Generate random values within the specified range for each axis
        random_x = np.random.uniform(working_envelope['min_x'], working_envelope['max_x'])
        random_y = np.random.uniform(working_envelope['min_y'], working_envelope['max_y'])
        random_z = np.random.uniform(working_envelope['min_z'], working_envelope['max_z'])

        # Create the random goal position as a NumPy array
        self.goal_position = np.array([random_x, random_y, random_z], dtype=np.float32)

        print(f'The goal position is: {self.goal_position}')

        # keep track of the number of steps
        self.steps = 0


    def reward_function(self, current_position, prev_position):
        distance_current = np.linalg.norm(current_position - self.goal_position)
        distance_prev = np.linalg.norm(prev_position - self.goal_position)
        distance = -0.1 * np.linalg.norm(current_position - self.goal_position)
        improvement = 0.5 * (distance_prev - distance_current)


        return (distance + improvement)
    

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        observation = self.sim.reset()

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
            
        working_envelope = {'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}

        # Generate random values within the specified range for each axis
        random_x = np.random.uniform(working_envelope['min_x'], working_envelope['max_x'])
        random_y = np.random.uniform(working_envelope['min_y'], working_envelope['max_y'])
        random_z = np.random.uniform(working_envelope['min_z'], working_envelope['max_z'])

        # Create the random goal position as a NumPy array
        self.goal_position = np.array([random_x, random_y, random_z], dtype=np.float32)

        robot_id = list(observation.keys())[0]

        observation_position = np.array(observation[robot_id]['pipette_position'], dtype=float)

        # Append the goal position to the pipette position
        observation = np.hstack((observation_position, self.goal_position), dtype=np.float32)
        
        # Reset the number of steps
        self.steps = 0

        info = {}

        return observation, info


    def step(self, action):
        
        terminated = False   

        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0.0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Because the simulation class expects a list of actions

        robot_id = list(observation.keys())[0]

        observation_position = np.array(observation[robot_id]['pipette_position'], dtype=float)

        # Append the goal position to the pipette position
        observation = np.hstack((observation_position, self.goal_position), dtype=np.float32)


        # Calculate the reward, this is something that you will need to experiment with to get the best results
        reward = self.reward_function(observation_position, self.previous_position)
        self.previous_position = observation_position

        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete.
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.

        distance_to_goal = np.linalg.norm(observation_position - self.goal_position)

        #print("This is distance to the goal " + str(distance_to_goal))

        if distance_to_goal <= 0.001:

            reward += 0.5 * abs(reward) * (1000//self.steps)

            print('terminated')

            terminated = True


        if self.steps > 1000:

            print(self.goal_position)

            truncated = True

        else:
            truncated = False

        info = {}  # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info


    def render(self, mode='human'):
        # Check if rendering is enabled
        if self.render_mode:
            # Implement the rendering logic here
            # For example, visualize the environment's state
            pass
        else:
            # If rendering is disabled, do nothing or print a message
            print("Rendering is disabled.")


    def get_plate_image(self):
        return self.sim.get_plate_image()


    def close(self):
        self.sim.close()
