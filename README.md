# SafarGo: Ride-Hailing Vehicle Repositioning Using Q-Learning

## Overview

This project presents an educational simulation of vehicle repositioning for a ride-hailing service called SafarGo. It explores how Reinforcement Learning (RL), particularly Q-learning, can help guide idle vehicles toward zones where passenger demand may be higher.

The example scenario considers 40 zones across Rawalpindi and Islamabad, with a fleet of 300 vehicles. A 10% pilot represents 30 vehicles.

These values are scenario settings for the assignment and should not be interpreted as verified operational data from an actual ride-hailing company.

## Problem Statement

Ride-hailing services may experience an imbalance between passenger demand and available vehicles. Some zones may have many idle vehicles, while other zones may have more ride requests than available drivers.

This project simulates how Q-learning can support vehicle repositioning decisions to improve service availability and reduce unnecessary movement.

## Objectives

- Simulate passenger demand and vehicle availability across different zones.
- Apply Q-learning to a simplified vehicle repositioning problem.
- Compare the learned strategy with baseline strategies included in the implementation.
- Evaluate simulation performance using available metrics.
- Provide an interactive interface for testing different simulation settings.

## Reinforcement Learning Components

### 1. Agent
The agent is the vehicle repositioning system that selects actions based on the current simulated situation.

### 2. Environment
The environment includes the simulated service zones, vehicle availability, passenger demand, and any additional factors represented in the code.

### 3. State
The state represents the current situation of the system. Depending on the implementation, it may include information about demand and available vehicle supply.

### 4. Actions
Actions represent the repositioning choices available to the agent in the simulation.

### 5. Reward Function
The reward function evaluates the outcome of an action. It can reward useful service outcomes and penalize undesirable outcomes, according to the formula implemented in the program.

### 6. Learning Algorithm
Q-learning helps the agent estimate the value of different actions. The agent uses rewards and estimated future rewards to improve its policy through repeated training.



## Main Features

Depending on the supplied implementation, the project may include:

- Simulation of ride demand and vehicle repositioning.
- Q-learning training and action selection.
- Comparison with simple baseline strategies.
- Tables and charts displaying generated simulation results.
- CSV exports for further analysis.

## Technologies Used

- Python
- Q-learning
- Reinforcement Learning
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Requirements

Install the required libraries using:

```bash
pip install streamlit numpy pandas matplotlib
```

If your program imports other libraries, install those as well.

## How to Run the Project

### Step 1: Open the Project

Open the project folder in VS Code or another suitable editor.

### Step 2: Install Dependencies

Run the installation command shown above in the terminal.

### Step 3: Start the Streamlit Application

If your application file is named `app.py`, execute:

```bash
streamlit run app.py
```

## Author Information

- **Haleema Sadia**



