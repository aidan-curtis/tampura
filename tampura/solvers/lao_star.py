import mdptoolbox.mdp
import numpy as np


class LAOStar:
    def __init__(self, transitions, reward, discount, epsilon=0.01, max_iter=1000):
        self.transitions = transitions
        self.reward = reward
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_states, self.n_actions = reward.shape
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)
        self.expanded = set()
        self.fringe = set()
        self.initial_state = None

    def run(self, initial_state):
        self.initial_state = initial_state
        self.expanded = set([initial_state])
        self.fringe = set()

        while True:
            # Expand the best partial solution
            self._expand_solution()

            # Perform value iteration on the expanded graph
            converged = self._value_iteration()

            if converged and len(self.fringe) == 0:
                break

        return self.V, self.policy

    def _expand_solution(self):
        new_fringe = set()
        for s in self.expanded:
            for a in range(self.n_actions):
                for s_next in range(self.n_states):
                    if self.transitions[a][s][s_next] > 0 and s_next not in self.expanded:
                        new_fringe.add(s_next)
        self.fringe.update(new_fringe)
        self.expanded.update(new_fringe)
        if not new_fringe:
            self.fringe.clear()  # Clear fringe if no new states are added

    def _value_iteration(self):
        for _ in range(self.max_iter):
            delta = 0
            for s in self.expanded:
                v = self.V[s]
                q = np.zeros(self.n_actions)

                for a in range(self.n_actions):
                    q[a] = self.reward[s, a] + self.discount * np.sum(
                        self.transitions[a, s] * self.V
                    )

                self.V[s] = np.max(q)
                self.policy[s] = np.argmax(q)
                delta = max(delta, abs(v - self.V[s]))

            if delta < self.epsilon:
                return True

        return False


def test_simple_mdp():
    # Simple 2-state MDP
    transitions = np.array(
        [[[0.7, 0.3], [0.3, 0.7]], [[0.2, 0.8], [0.9, 0.1]]]  # Action 0  # Action 1
    )
    rewards = np.array([[0, 0], [0, 1]])  # Rewards for state 0  # Rewards for state 1
    gamma = 0.9

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    print("Simple MDP Test:")
    print("Value Iteration V:", vi.V)

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = lao.run(initial_state=0)

    print("LAO* V:", lao_v)
    print("Value Iteration Policy:", vi.policy)
    print("LAO* Policy:", lao_policy)
    print()


def test_grid_world():
    # 3x3 Grid World MDP
    n_states = 9
    n_actions = 4  # Up, Right, Down, Left

    transitions = np.zeros((n_actions, n_states, n_states))
    rewards = np.zeros((n_states, n_actions))

    # Set up transitions (with 0.1 probability of failing and staying in place)
    for s in range(n_states):
        for a in range(n_actions):
            next_s = s
            if a == 0 and s >= 3:  # Up
                next_s = s - 3
            elif a == 1 and s % 3 < 2:  # Right
                next_s = s + 1
            elif a == 2 and s < 6:  # Down
                next_s = s + 3
            elif a == 3 and s % 3 > 0:  # Left
                next_s = s - 1

            transitions[a, s, next_s] = 0.9
            transitions[a, s, s] += 0.1

    # Set rewards (goal state is 8)
    rewards[8, :] = 1  # Reward for being in state 8 (goal state)

    gamma = 0.9

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = lao.run(initial_state=0)

    print("Grid World Test:")
    print("Value Iteration V:", vi.V)
    print("LAO* V:", lao_v)
    print("Value Iteration Policy:", vi.policy)
    print("LAO* Policy:", lao_policy)
    print()


def test_unreachable_states():
    # MDP with unreachable states
    transitions = np.array(
        [
            [[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]],  # Action 0
            [[0.8, 0.2, 0], [0, 1, 0], [0, 0, 1]],  # Action 1
        ]
    )
    rewards = np.array(
        [
            [0, 0],  # Rewards for state 0
            [1, 1],  # Rewards for state 1
            [2, 2],  # Rewards for state 2
        ]
    )
    gamma = 0.9

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = lao.run(initial_state=0)

    print("Unreachable States Test:")
    print("Value Iteration V:", vi.V)
    print("LAO* V:", lao_v)
    print("Value Iteration Policy:", vi.policy)
    print("LAO* Policy:", lao_policy)
    print()


def run_lao_star_with_timeout(lao, initial_state, max_iterations=1000):
    lao.run(initial_state)
    return lao.V, lao.policy


def test_large_grid_world():
    # 10x10 Grid World MDP
    n_states = 100
    n_actions = 4  # Up, Right, Down, Left

    transitions = np.zeros((n_actions, n_states, n_states))
    rewards = np.zeros((n_states, n_actions))

    # Set up transitions (with 0.1 probability of failing and staying in place)
    for s in range(n_states):
        for a in range(n_actions):
            next_s = s
            if a == 0 and s >= 10:  # Up
                next_s = s - 10
            elif a == 1 and s % 10 < 9:  # Right
                next_s = s + 1
            elif a == 2 and s < 90:  # Down
                next_s = s + 10
            elif a == 3 and s % 10 > 0:  # Left
                next_s = s - 1

            transitions[a, s, next_s] = 0.9
            transitions[a, s, s] += 0.1

    # Set rewards (multiple goal states)
    goal_states = [33, 66, 99]
    for goal in goal_states:
        rewards[goal, :] = 1  # Reward for reaching goal states

    # Add small negative reward for each step
    rewards[rewards == 0] = -0.01

    gamma = 0.99

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = run_lao_star_with_timeout(lao, initial_state=0)

    print("Large Grid World Test:")
    print("Value Iteration V (first 10 states):", vi.V[:10])
    print("LAO* V (first 10 states):", lao_v[:10])
    print("Value Iteration Policy (first 10 states):", vi.policy[:10])
    print("LAO* Policy (first 10 states):", lao_policy[:10])
    print()


def test_stochastic_wind_grid():
    # 5x5 Grid World with stochastic wind
    n_states = 25
    n_actions = 4  # Up, Right, Down, Left

    transitions = np.zeros((n_actions, n_states, n_states))
    rewards = np.zeros((n_states, n_actions))

    # Set up transitions with stochastic wind
    for s in range(n_states):
        for a in range(n_actions):
            intended_s = s
            if a == 0 and s >= 5:  # Up
                intended_s = s - 5
            elif a == 1 and s % 5 < 4:  # Right
                intended_s = s + 1
            elif a == 2 and s < 20:  # Down
                intended_s = s + 5
            elif a == 3 and s % 5 > 0:  # Left
                intended_s = s - 1

            # 70% chance of going in the intended direction
            transitions[a, s, intended_s] = 0.7

            # 30% chance of being blown by the wind (up or down)
            wind_up = max(0, s - 5)
            wind_down = min(24, s + 5)
            transitions[a, s, wind_up] += 0.15
            transitions[a, s, wind_down] += 0.15

    # Set rewards (goal state is 24)
    rewards[24, :] = 10  # High reward for reaching the goal
    rewards[0, :] = -10  # Penalty for going back to start

    # Add small negative reward for each step
    rewards[rewards == 0] = -0.1

    gamma = 0.95

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = run_lao_star_with_timeout(lao, initial_state=0)

    print("Stochastic Wind Grid Test:")
    print("Value Iteration V:", vi.V)
    print("LAO* V:", lao_v)
    print("Value Iteration Policy:", vi.policy)
    print("LAO* Policy:", lao_policy)
    print()


def test_maze_with_traps():
    # 6x6 Maze with traps and multiple goals
    n_states = 36
    n_actions = 4  # Up, Right, Down, Left

    transitions = np.zeros((n_actions, n_states, n_states))
    rewards = np.zeros((n_states, n_actions))

    # Set up transitions
    for s in range(n_states):
        for a in range(n_actions):
            next_s = s
            if a == 0 and s >= 6:  # Up
                next_s = s - 6
            elif a == 1 and s % 6 < 5:  # Right
                next_s = s + 1
            elif a == 2 and s < 30:  # Down
                next_s = s + 6
            elif a == 3 and s % 6 > 0:  # Left
                next_s = s - 1

            transitions[a, s, next_s] = 1.0

    # Add walls (transitions stay in the same state)
    walls = [7, 8, 13, 14, 19, 20, 25, 26]
    for wall in walls:
        for a in range(n_actions):
            transitions[a, wall] = np.zeros(n_states)
            transitions[a, wall, wall] = 1.0

    # Set rewards
    goal_states = [17, 35]  # Multiple goals
    trap_states = [5, 23, 31]  # Traps

    for goal in goal_states:
        rewards[goal, :] = 10  # High reward for reaching goals
    for trap in trap_states:
        rewards[trap, :] = -10  # Penalty for falling into traps

    # Add small negative reward for each step
    rewards[rewards == 0] = -0.1

    gamma = 0.99

    # Value Iteration
    vi = mdptoolbox.mdp.ValueIteration(transitions, rewards, gamma)
    vi.run()

    # LAO*
    lao = LAOStar(transitions, rewards, gamma)
    lao_v, lao_policy = run_lao_star_with_timeout(lao, initial_state=0)

    print("Maze with Traps Test:")
    print("Value Iteration V:", vi.V)
    print("LAO* V:", lao_v)
    print("Value Iteration Policy:", vi.policy)
    print("LAO* Policy:", lao_policy)
    print()
