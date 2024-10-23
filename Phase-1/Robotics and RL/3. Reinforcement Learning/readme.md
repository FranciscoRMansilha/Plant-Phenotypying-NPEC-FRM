# 3. Training and using a Reinforcement Learning Model
---

## What is a Reinforcement Learning Controller? 

Reinforcement learning is like teaching the robot through trial and error. The robot tries different actions, and when it does something right—like reaching the correct spot for inoculation—it gets rewarded. Over time, it learns which actions are best to achieve the goal. It's similar to how we learn: if we try something and it works, we're more likely to do it again. In this project, I trained a RL model that used an algorithm called PPO (Proximal Policy Optimization).



## An example of how the PID controller looks like in action:

<img src='plots/RL controller.gif'>

---

## Code Documentation

The code used to setup the simulation and inoculation with the PID controller can be found in `RL_controller.ipynb`. The code used to train, monitor and test the RL model can be found in `RL_PPO_training.py` and `Testing_PPO_model.py`.

### Dependencies

The code utilizes several libraries for its operations:

- `stable_baselines3` for the RL model.
- `ot2_gym_wrapper` for a custom environment that simulates the robotic arm's movements.
- `matplotlib` for plotting the results.
- `simple_pid` for the PID control.
- `pandas` and `numpy` for data handling and calculations.
- `time` and `os` for timing executions and handling file paths.

### Execution Flow

1. **Environment Initialization:** A custom environment, `OT2Env`, is loaded to simulate the robotic arm's actions, with 'human' rendering to visualize the movements.

```python
from ot2_gym_wrapper import OT2Env 

# instantiate your custom environment
env = OT2Env(render='human') 
```

2. **Image Processing:** An image of the plate is obtained, and landmarks (specifically, the primary root tips of plants) are detected. These coordinates are then adjusted based on a conversion factor that translates pixel measurements into real-world distances, taking into account the plate's dimensions.

```python
# Obtaining the image in the plate
image_path = env.get_plate_image()

# Applying landmark detection for the primary root tip
coordinates = landmarks(image_path)
```

3. **Goal Position Adjustment:** For each detected plant, the goal position is calculated by adjusting the detected coordinates with a conversion factor and additional offsets to align with the robotic arm's coordinate system.

```python
#Creating Conversion factor
plate_size_m = 0.15

plate_size_pixels = 2804

conversion_factor = plate_size_m / plate_size_pixels

# Empty list to hold adjusted coordinates
goal_positions = []

# Additional offset or adjustment for each axis
offset_x = 0.10775
offset_y = 0.088

# Loop through each dictionary in the coordinates list
for coord in coordinates:
    # Extract the primary_root_tip tuple
    original_position = coord["primary_root_tip"]
    # Adjust the tuple values by multiplying with the conversion factor and round the results
    adjusted_position = (original_position[1] * conversion_factor, original_position[0] * conversion_factor)
    # Append the adjusted coordinate tuple to the goal_positions list
    goal_positions.append(adjusted_position)

# Print the resulting list of adjusted positions
print(goal_positions)
```

4. **Reinforcement Learning Control:** A pre-trained RL model (specifically, a PPO model) is loaded and used to guide the robotic arm towards each goal position. The distance to the goal is monitored, and the inoculation is performed once the arm is within a predefined error threshold. The distances over time for each plant are plotted and saved.

```python
# Load the trained agent
model = PPO.load('RL_models_2/rl_model_9')

# Directory for saving plots
plots_directory = 'plots_dataframes'

# Data structure to hold the data for DataFrame
plant_data = []

# Resetting the environment
obs, info = env.reset()

x_adjustment = 0.014
y_adjustment = 0

# Add a counter for the goal positions
for index, goal_pos in enumerate(goal_positions, start=1):  # Start=1 to begin counting from 1

    # Initialize a list to store distances for plotting
    distances = []

    # Set the goal position for the robot
    goal_pos_x = 0.10775 + goal_pos[0] + x_adjustment
    goal_pos_y = 0.088 + goal_pos[1] + y_adjustment
    goal_pos_z = 0.1695
    goal_pos = np.array([goal_pos_x, goal_pos_y, goal_pos_z])
    env.goal_position = goal_pos
    
    # Modify the print statement to include the goal position number
    print(f'\nPlant {index}')  # Use the counter here
    print(f'Goal position is: {goal_pos_x, goal_pos_y, goal_pos_z}')

    start_time = time.time()
       
    # Run the control algorithm until the robot reaches the goal position
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        
        # calculate the distance between the pipette and the goal
        distance = obs[3:] - obs[:3]  # goal position - pipette position
        error = np.linalg.norm(distance)
        
        # Store the error for plotting
        distances.append(error)
        
        # Drop the inoculum if the robot is within the required error
        if error < 0.009:
            action = np.array([0, 0, 0, 1])
            obs, rewards, terminated, truncated, info = env.step(action)
            end_time = time.time()  # Stop the timer
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f'Complete with an error of \n{error} in {elapsed_time:.2f} seconds and {i} iterations')
            break

        if terminated:
            print('Episode terminated')
            obs, info = env.reset()
            print('Environment reset')
            break

    # After the loop, save the plot to a file
    plt.figure(figsize=(10, 5))
    plt.plot(distances, label=f'Plant {index}')
    plt.xlabel('Step')
    plt.ylabel('Distance to Goal')
    plt.title(f'Distance to Goal Over Time for Plant {index}')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(plots_directory, f'plot_plant_{index}_RL.png')
    plt.savefig(plot_filename)

    # Add the data to the list for DataFrame
    plant_data.append({'plant': index, 'error': error, 'time': elapsed_time})

# Create a DataFrame from the collected data
df_plants = pd.DataFrame(plant_data)

# Save the DataFrame to a CSV file if needed
df_plants.to_csv('plots_dataframes/data_RL.csv', index=False)
```