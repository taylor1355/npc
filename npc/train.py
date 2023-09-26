from agilerl.utils.utils import initialPopulation, makeVectEnvs
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train
from agilerl.hpo.mutation import Mutations
import torch

def create_mutation_obj(INIT_HP, MUTATION_PARAMS, NET_CONFIG, device):
    mutations = Mutations(
        algo=INIT_HP['ALGO'],
        no_mutation=MUTATION_PARAMS['NO_MUT'],
        architecture=MUTATION_PARAMS['ARCH_MUT'],
        new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],
        parameters=MUTATION_PARAMS['PARAMS_MUT'],
        activation=MUTATION_PARAMS['ACT_MUT'],
        rl_hp=MUTATION_PARAMS['RL_HP_MUT'],
        rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],
        mutation_sd=MUTATION_PARAMS['MUT_SD'],
        min_lr=MUTATION_PARAMS['MIN_LR'],
        max_lr=MUTATION_PARAMS['MAX_LR'],
        min_learn_step=MUTATION_PARAMS['MIN_LEARN_STEP'],
        max_learn_step=MUTATION_PARAMS['MAX_LEARN_STEP'],
        min_batch_size=MUTATION_PARAMS['MIN_BATCH_SIZE'],
        max_batch_size=MUTATION_PARAMS['MAX_BATCH_SIZE'],
        arch=NET_CONFIG['arch'],
        rand_seed=MUTATION_PARAMS['RAND_SEED'],
        device=device
    )


# TODO (first to last):
# - Replace PPO with custom simplified implementation
# - Replace LunarLander with custom frozen_lake and choose input scheme s.t. observations are text (ie from thought stream)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Actor hidden size
    }

    INIT_HP = {
        "ALGO": 'PPO',                  #
        "BATCH_SIZE": 128,              #
        "LR": 1e-3,                     #
        'EPISODES': 2000,               # Max no. episodes
        'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                  # Discount factor
        'GAE_LAMBDA': 0.95,             # Lambda for general advantage estimation
        'ACTION_STD_INIT': 0.6,         # Initial action standard deviation
        'CLIP_COEF': 0.2,               # Surrogate clipping coefficient
        'ENT_COEF': 0.01,               # Entropy coefficient
        'VF_COEF': 0.5,                 # Value function coefficient
        'MAX_GRAD_NORM': 0.5,           # Maximum norm for gradient clipping
        'TARGET_KL': None,              # Target KL divergence threshold
        'UPDATE_EPOCHS': 4,             # Number of policy update epochs
        'LEARN_STEP': 1,                # Learning frequency
        "POPULATION_SIZE": 4,           #
        'TOURN_SIZE': 2,                #
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'DISCRETE_ACTIONS': True,       #
        'WANDB': True                   # Log with Weights and Biases
    }

    MUTATION_PARAMS = {
        "NO_MUT": 0.4,                              # No mutation
        "ARCH_MUT": 0.2,                            # Architecture mutation
        "NEW_LAYER": 0.2,                           # New layer mutation
        "PARAMS_MUT": 0.2,                           # Network parameters mutation
        "ACT_MUT": 0,                               # Activation layer mutation
        "RL_HP_MUT": 0.2,                           # Learning HP mutation
        # Learning HPs to choose from
        "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
        "MUT_SD": 0.1,                              # Mutation strength
        "RAND_SEED": 42,                            # Random seed
        "MIN_LR": 0.0001,                           # Define max and min limits for mutating RL hyperparams
        "MAX_LR": 0.01,
        "MIN_LEARN_STEP": 1,
        "MAX_LEARN_STEP": 200,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024
    }

    env_name = "LunarLander-v2"
    env = makeVectEnvs(env_name, num_envs=8)  # Create environment
    state_dim = env.single_observation_space.shape  # Continuous observation space
    one_hot = False  # Does not require one-hot encoding
    action_dim = env.single_action_space.n  # Discrete action space

    agent_pop = initialPopulation(
        algo=INIT_HP['ALGO'],  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    mutations = create_mutation_obj(INIT_HP, MUTATION_PARAMS, NET_CONFIG, device)    

    trained_pop, pop_fitnesses = train(
        env=env,                                 # Gym-style environment
        env_name=env_name,                       # Environment name
        algo=INIT_HP['ALGO'],                    # Algorithm
        pop=agent_pop,                           # Population of agents
        INIT_HP=INIT_HP,                         #
        MUT_P=MUTATION_PARAMS,                   #
        n_episodes=INIT_HP['EPISODES'],          # Max number of training episodes
        evo_epochs=INIT_HP['EVO_EPOCHS'],        # Evolution frequency
        evo_loop=1,                              # Number of evaluation episodes per agent
        target=INIT_HP['TARGET_SCORE'],          # Target score for early stopping
        tournament=tournament,                   # Tournament selection object
        mutation=mutations,                      # Mutations object
        wb=INIT_HP['WANDB']                      # Weights and Biases tracking
    )
    
if __name__ == '__main__':
    main()