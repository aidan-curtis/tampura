from tampura.config.config import register_planner

from .contingent_policy import ContingentPolicy
from .dqn_policy import DQNPolicy
from .policy import Policy, RolloutHistory
from .random_policy import RandomPolicy
from .replay_policy import ReplayPolicy
from .tabular_q_learning import TabularQLearning
from .tampura_policy import TampuraPolicy
from .uct_policy import UCTPolicy

register_planner("random_policy", RandomPolicy)
register_planner("uct_policy", UCTPolicy)
register_planner("tabular_q_learning", TabularQLearning)
register_planner("tampura_policy", TampuraPolicy)
register_planner("replay_policy", ReplayPolicy)
register_planner("dqn_policy", DQNPolicy)
register_planner("contingent_policy", ContingentPolicy)
