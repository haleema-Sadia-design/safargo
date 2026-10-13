import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# SafarGo Vehicle Repositioning Simulation
ZONES = 40
VEHICLES = 300
PILOT_PERCENT = 0.10
PILOT_VEHICLES = int(VEHICLES * PILOT_PERCENT)
INTERVALS_PER_DAY = 288

ALPHA = 0.5
GAMMA = 0.9
EPSILON = 1.0
MIN_EPSILON = 0.05
TRAINING_EPISODES = 500
LAMBDA = 1.0


def generate_demand(rng):
    """Generate simulated passenger demand for 40 zones."""
    hour = rng.randrange(24)

    # Simulated demand patterns, not real SafarGo data.
    morning_peak = 1.5 if 7 <= hour <= 10 else 1.0
    evening_peak = 1.6 if 16 <= hour <= 20 else 1.0

    base = np.array([
        rng.uniform(2, 8) for _ in range(ZONES)
    ])

    if 7 <= hour <= 10 or 16 <= hour <= 20:
        base *= max(morning_peak, evening_peak)

    return np.array([
        rng.poisson(value) for value in base
    ])


def get_state(demand, supply):
    demand_level = int(np.mean(demand) // 3)
    imbalance = int(np.mean(demand - supply) > 0)
    return min(demand_level, 2) * 2 + imbalance


def train_agent(seed=42):
    rng = random.Random(seed)
    q_table = np.zeros((6, 2))
    rewards = []
    epsilon = EPSILON

    for episode in range(TRAINING_EPISODES):
        total_reward = 0

        for _ in range(INTERVALS_PER_DAY):
            demand_level = rng.randint(0, 2)
            imbalance = rng.randint(0, 1)
            state = demand_level * 2 + imbalance

            if rng.random() < epsilon:
                action = rng.randint(0, 1)
            else:
                action = int(np.argmax(q_table[state]))

            # Illustrative training feedback:
            # reposition when demand pressure is high.
            if action == 1:
                reward = (
                    3 * demand_level
                    + 2 * imbalance
                    - 1.5
                )
            else:
                reward = (
                    1.5
                    - 2 * demand_level
                    - imbalance
                )

            next_demand = rng.randint(0, 2)
            next_imbalance = rng.randint(0, 1)
            next_state = next_demand * 2 + next_imbalance

            q_table[state, action] += ALPHA * (
                reward
                + GAMMA * np.max(q_table[next_state])
                - q_table[state, action]
            )

            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * 0.99)

    return q_table, rewards


def reposition(vehicles, demand, policy, q_table, rng):
    """Move only a limited number of vehicles between zones."""
    moved = 0
    available = vehicles.copy()

    for _ in range(PILOT_VEHICLES):
        source_candidates = np.where(
            available > np.maximum(1, demand * 0.4)
        )[0]

        if len(source_candidates) == 0:
            break

        source = int(rng.choice(list(source_candidates)))

        if policy == "Random":
            destination = rng.randrange(ZONES)

        else:
            scores = demand - available

            if policy == "Busiest-Zone":
                destination = int(np.argmax(scores))

            else:
                state = get_state(demand, available)
                action = int(np.argmax(q_table[state]))

                if action == 0:
                    break

                destination = int(np.argmax(scores))

        if destination == source:
            continue

        if available[source] <= 1:
            continue

        available[source] -= 1
        available[destination] += 1
        moved += 1

    return available, moved


def simulate(policy, q_table, seed=123, days=5):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # Same starting fleet distribution for each controller.
    initial = np.full(ZONES, VEHICLES // ZONES)
    initial[:VEHICLES % ZONES] += 1

    vehicles = initial.copy()
    total_completed = 0
    total_waiting = 0
    total_fuel = 0
    total_cancelled = 0
    zone_service = np.zeros(ZONES)
    total_reward = 0
    records = []

    for day in range(days):
        for interval in range(INTERVALS_PER_DAY):
            demand = rng.poisson(
                np.array([
                    rng.uniform(2, 8) for _ in range(ZONES)
                ])
            )

            if policy != "Baseline":
                vehicles, moved = reposition(
                    vehicles, demand, policy, q_table, py_rng
                )
            else:
                moved = 0

            completed_by_zone = np.minimum(demand, vehicles)
            completed = int(completed_by_zone.sum())

            waiting = int(np.maximum(
                demand - vehicles, 0
            ).sum())

            cancelled = int(waiting * 0.2)
            fuel = moved * 1.0

            service = np.divide(
                completed_by_zone,
                np.maximum(demand, 1)
            )

            unfairness = float(np.std(service))

            # Illustrative reward based on simulated metrics.
            revenue = completed / VEHICLES
            reward = (
                2 * completed
                + LAMBDA * revenue
                - 0.5 * waiting
                - fuel
                - 3 * cancelled
                - 2 * unfairness
            )

            # Vehicles used for completed rides become available
            # again in the simplified simulation.
            total_completed += completed
            total_waiting += waiting
            total_fuel += fuel
            total_cancelled += cancelled
            zone_service += completed_by_zone
            total_reward += reward

            records.append({
                "Day": day + 1,
                "Interval": interval + 1,
                "Controller": policy,
                "Completed Rides": completed,
                "Waiting Requests": waiting,
                "Fuel Units": fuel,
                "Cancelled Rides": cancelled,
                "Reward": reward
            })

    return {
        "Controller": policy,
        "Completed Rides": total_completed,
        "Average Waiting Requests": total_waiting / (
            days * INTERVALS_PER_DAY
        ),
        "Fuel Consumption (units)": total_fuel,
        "Cancelled Rides": total_cancelled,
        "Average Reward": total_reward / (
            days * INTERVALS_PER_DAY
        ),
        "Zone Service Std": np.std(zone_service)
    }, pd.DataFrame(records)


st.set_page_config(
    page_title="SafarGo RL Simulation",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 SafarGo Vehicle Repositioning Using Q-Learning")

st.write(
    "Simulate vehicle repositioning across 40 zones. "
    "The simulation uses 300 vehicles and limits each "
    "repositioning decision to at most 30 vehicles."
)

st.warning(
    "Educational simulation only. These are not actual "
    "SafarGo operational results."
)

st.sidebar.header("Simulation Settings")
days = st.sidebar.selectbox(
    "Evaluation duration (days)", [1, 3, 5, 7], index=2
)
seed = st.sidebar.number_input(
    "Random seed", min_value=1, value=42
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Zones", ZONES)
c2.metric("Total Vehicles", VEHICLES)
c3.metric("Pilot Limit", PILOT_VEHICLES)
c4.metric("Decision Interval", "5 minutes")

if st.button("Train Q-Learning and Generate Graphs",
              type="primary"):

    with st.spinner("Training and simulating controllers..."):
        q_table, training_rewards = train_agent(int(seed))

        results = []
        histories = []

        for policy in [
            "Baseline", "Random",
            "Busiest-Zone", "Q-Learning"
        ]:
            metrics, history = simulate(
                policy,
                q_table,
                seed=int(seed) + 100,
                days=int(days)
            )
            results.append(metrics)
            histories.append(history)

        results_df = pd.DataFrame(results)
        history_df = pd.concat(histories, ignore_index=True)

        st.session_state["q_table"] = q_table
        st.session_state["training_rewards"] = training_rewards
        st.session_state["results"] = results_df
        st.session_state["history"] = history_df

if "results" in st.session_state:
    results = st.session_state["results"]
    rewards = st.session_state["training_rewards"]
    history = st.session_state["history"]

    st.subheader("Performance Results")
    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True
    )

    graph_columns = [
        ("Completed Rides", "Completed Rides"),
        ("Average Waiting Requests", "Average Waiting Requests"),
        ("Fuel Consumption (units)", "Fuel Consumption"),
        ("Average Reward", "Average Reward"),
        ("Cancelled Rides", "Cancelled Rides"),
        ("Zone Service Std", "Zone Service Variation")
    ]

    for column, title in graph_columns:
        st.subheader(title)
        fig, ax = plt.subplots()
        ax.bar(results["Controller"], results[column])
        ax.set_ylabel(column)
        ax.set_title(title + " by Controller")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Q-Learning Training Reward")
    fig, ax = plt.subplots()
    ax.plot(rewards, alpha=0.35, label="Episode reward")
    ax.plot(
        pd.Series(rewards).rolling(
            20, min_periods=1
        ).mean(),
        label="20-episode average"
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Q-Table")
    q_df = pd.DataFrame(
        st.session_state["q_table"],
        columns=["Stay", "Reposition"]
    )
    q_df.index.name = "State"
    st.dataframe(q_df.round(3), use_container_width=True)

    st.download_button(
        "Download Performance Results CSV",
        results.to_csv(index=False),
        "safargo_results.csv",
        "text/csv"
    )

    st.download_button(
        "Download Simulation History CSV",
        history.to_csv(index=False),
        "safargo_history.csv",
        "text/csv"
    )

st.caption(
    "Results depend on the simulation assumptions, random seed, "
    "reward design, and simplified vehicle model."
)