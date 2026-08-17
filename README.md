# Reinforcement Learning Taxi Agent

This repository contains a Jupyter notebook that trains a Q-learning policy for Gymnasium's `Taxi-v3` environment, uses the learned Q-table to create labelled state/action data, and trains a scikit-learn multilayer perceptron to imitate that policy.

## Features

- Train a tabular Q-learning agent with configurable alpha, gamma and epsilon values.
- Apply an additional visit penalty during Q-value updates.
- Plot episode rewards during Q-learning training.
- Save and load the learned Q-table with `pickle`.
- Generate labelled state/action samples from the learned Q-table.
- Train an `MLPClassifier` on the generated state/action dataset.
- Report accuracy, weighted F1 score, a confusion matrix and per-class accuracy.
- Evaluate the MLP policy over multiple Taxi-v3 episodes and plot reward/step totals.
- Render Taxi-v3 with Gymnasium's human render mode.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python |
| Environment | Gymnasium `Taxi-v3` |
| Reinforcement learning | NumPy Q-table implementation |
| Neural network | scikit-learn `MLPClassifier` |
| Visualisation | Matplotlib |
| Format | Jupyter Notebook |

## Requirements

The repository has no `requirements.txt` or other dependency manifest. The notebook imports NumPy, Gymnasium, scikit-learn and Matplotlib.

## Running

```bash
jupyter notebook "APPLIED ROBOTICS.ipynb"
```

Run the notebook cells in order. The main training cell creates `Taxi-v3` with `render_mode="human"`, trains for 3,000 episodes, writes `q_table.pkl`, trains the MLP and then evaluates the MLP policy.

The previous README referred to `taxi_agent.py`, `requirements.txt` and `example_plot.png`; none of those files exists in this repository, so those instructions have been removed.

## Testing

There is no separate automated test suite. Model evaluation is performed inside the notebook using classification metrics and Taxi-v3 evaluation episodes.

## Licence

MIT. See [LICENSE](LICENSE).
