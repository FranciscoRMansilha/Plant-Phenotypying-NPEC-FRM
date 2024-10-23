# 1. Set-up a virtual representation of Opentrons OT-2

The OT-2 Digital Twin is a virtual representation of the Opentrons OT-2, a popular robotic liquid handling system used in labs. To use the OT-2 Digital Twin with PyBullet, you need to follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:

   ```bash
   git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git
   ```

4. Once the repository is cloned, you can copy its contents to your project directory.

1. **Initialize the Simulation Environment:**
   
   Navigate to the directory where you copied the contents of the repository. in the same folder as the `sim_class.py` file, create a new Python file and import the `Simulation` class.

   You then need to create an instance of the `Simulation` class. The constructor takes `num_agents` as a parameter, which specifies the number of robotic agents in the simulation.

   ```python
   from sim_class import Simulation

   # Initialize the simulation with a specified number of agents
   sim = Simulation(num_agents=1)  # For one robot
   ```

2. **Sending Commands to the Robot:**

   To control the robot, you can use the `run` method of the `Simulation` class. This method takes `actions` and `num_steps` as parameters. `actions` is a list of commands for each joint of the robot, and `num_steps` is the number of simulation steps to execute. 

   ```python
   # Example action: Move joints with specific velocities
   velocity_x = 0.1
   velocity_y = 0.1     
   velocity_z = 0.1
   drop_command = 0
   actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

   # Run the simulation for a specified number of steps
   sim.run(actions, num_steps=1000)
   ```

## Working Envelope:

These values can be obtained by running the jupyter notebook `working_envelope.ipynb` where you can find documented code for running a simulation in which commands are sent to the robot and observations are returned.

- min_x: -0.1871

- max_x: 0.253

- min_y: -0.1706 

- max_y: 0.2195

- min_z: 0.1197

- max_z: 0.2898

{'min_x': -0.1871, 'max_x': 0.253, 'min_y': -0.1706, 'max_y': 0.2195, 'min_z': 0.1197, 'max_z': 0.2898}