# Reinforcement Learning Taxi Agent

This repository contains an implementation of a **Reinforcement Learning (Q-Learning)** algorithm and a **Multilayer Perceptron (MLP)** for controlling a robot taxi agent in a simulated environment. The project uses the `Taxi-v3` environment provided by the Gymnasium API and demonstrates how an agent learns to efficiently pick up and drop off passengers while avoiding illegal actions and minimizing steps.

## Description

The project addresses the **Taxi Problem**, where the objective is for a robot taxi to:
1. Navigate a 5x5 grid environment.
2. Pick up a passenger from a designated location.
3. Drop off the passenger at their specified destination.

The solution is divided into two main components:
1. **Q-Learning**: A reinforcement learning approach to train the agent by learning optimal Q-values based on rewards and penalties.
2. **Neural Network Control**: A Multilayer Perceptron (MLP) trained on the Q-learning data to predict the next optimal action for the taxi.

## Features

- **Q-Learning Implementation**: 
  - Custom implementation of the Q-learning algorithm.
  - Hyperparameters such as learning rate, discount factor, and exploration rate are tunable for optimization.

- **MLP-Based Control**:
  - Uses the `sklearn` library to train a neural network to predict the agent's actions.
  - Supports performance evaluation using metrics like accuracy, F1-score, and confusion matrices.

- **Visualizations**:
  - Tracks rewards and steps over episodes.
  - Provides insights into training progress through plots and cumulative metrics.

- **Policy Evaluation**:
  - Evaluates the trained Q-learning and MLP policies over multiple episodes.
  - Includes graphical rendering of the taxi's actions in the environment.

## Technologies Used

- **Python**: Main programming language.
- **Gymnasium API**: For the `Taxi-v3` environment simulation.
- **Scikit-learn**: For the implementation of the Multilayer Perceptron.
- **Matplotlib**: For plotting metrics and training results.

## Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Reinforcement-Learning-Taxi-Agent.git
    cd Reinforcement-Learning-Taxi-Agent
    ```

2. **Install Dependencies**:
    Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**:
    - Train the Q-learning agent and evaluate its performance:
      ```bash
      python taxi_agent.py
      ```

4. **Customize Parameters**:
    - Modify parameters such as `gamma`, `alpha`, and `epsilon` in the script to experiment with the learning behavior.

## Example Output

- **Training Metrics**:
    ```
    Episode 1: Total Reward: -875, Steps Taken: 200
    Episode 10: Total Reward: -596, Steps Taken: 200
    ...
    ```

- **Visualizations**:
    - Plot of total rewards and steps taken per episode:

    ![Training Performance](example_plot.png)

- **Environment Render**:
    Example of the taxi environment during evaluation:
    ```
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
    ```

## Future Enhancements

- Experiment with advanced neural network architectures (e.g., Convolutional Neural Networks).
- Implement additional reward structures for better performance.
- Extend the project to continuous environments using Proximal Policy Optimization (PPO).

## License

This project is licensed under the MIT License.

## Acknowledgments

- The `Gymnasium API` for providing the `Taxi-v3` environment.
- Reinforcement learning concepts adapted from Sutton and Barto's *Reinforcement Learning: An Introduction*.
