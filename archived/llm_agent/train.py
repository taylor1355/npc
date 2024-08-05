from .prompt_util import create_prompt
from .frozen_lake import FrozenLake
from .base_agent import BaseAgent
from .llm_agent import LLMAgent
from .llm_args import LLMArgs
from .ppo import PPO

from agilerl.utils.utils import makeVectEnvs
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train
from agilerl.hpo.mutation import Mutations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange

def create_mutation_obj(INIT_HP, MUTATION_PARAMS, device):
    return Mutations(
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
        arch='mlp', # TODO: create a custom mutations class which does not take this argument
        rand_seed=MUTATION_PARAMS['RAND_SEED'],
        device=device
    )

def train_agent_one_episode(agent, env, max_steps):
    llm_agent = agent.agent

    state = env.reset()[0]  # Reset environment at start of episode
    reward = 0
    idx_step = 0
    internal_state = llm_agent.update(state, reward, idx_step, env.thought_prompt_factory, env.state_to_str)
    score = 0
    
    internal_states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    
    # TODO: keep track of idx_step inside env instead of in the training loop
    # TODO: pass env to agent.update() instead of idx_step, env.thought_prompt_factory and env.state_to_str}

    for idx_step in range(max_steps):
        # Get next action from agent
        # TODO: internal state should include an attention mask
        action, log_prob, _, value = agent.getAction(internal_state)
        action = action.item()
        state, reward, done, trunc, _ = env.step(
            action
        )  # Act in environment
        next_internal_state = llm_agent.update(state, reward, idx_step, env.thought_prompt_factory, env.state_to_str)

        internal_states.append(next_internal_state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
    
        internal_state = next_internal_state
        score += reward

        if done:
            break
    
    agent.scores.append(score)
    
    experiences = (
        internal_states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_internal_state,
    )
    # Learn according to agent's RL algorithm
    agent.learn(experiences)
    
    agent.steps[-1] += idx_step + 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INIT_HP = {
        "ALGO": 'PPO',                  #
        "BATCH_SIZE": 128,              #
        "LR": 1e-3,                     #
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
        "POPULATION_SIZE": 3,           #
        'TOURN_SIZE': 2,                #
        'POLICY_FREQ': 2,               # Policy network update frequency
        'DISCRETE_ACTIONS': True,       #
        'WANDB': True                   # Log with Weights and Biases
    }

    MUTATION_PARAMS = {
        "NO_MUT": 0.4,                              # No mutation
        "ARCH_MUT": 0,                              # Architecture mutation
        "NEW_LAYER": 0,                             # New layer mutation
        "PARAMS_MUT": 0,                            # Network parameters mutation
        "ACT_MUT": 0,                               # Activation layer mutation
        "RL_HP_MUT": 0.6,                           # Learning HP mutation
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

    #env_name = "LunarLander-v2"
    #env = makeVectEnvs(env_name, num_envs=8)
    env = FrozenLake()
    # state_dim = env.single_observation_space.shape  # Continuous observation space
    # one_hot = False  # Does not require one-hot encoding
    # action_dim = env.single_action_space.n  # Discrete action space

    agent_pop = []
    for i in range(INIT_HP["POPULATION_SIZE"]):
        # flattened_state_dim = np.prod(state_dim)


        model_path = "princeton-nlp/Sheared-LLaMA-1.3B"#Open-Orca/oo-phi-1_5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        tokenizer.pad_token = tokenizer.eos_token

        #model_path = "Open-Orca/oo-phi-1_5"
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #llm = AutoModelForCausalLM.from_pretrained(
        #    model_path,
        #    trust_remote_code=True,
        #    torch_dtype=torch.bfloat16
        #).to(device)
        #tokenizer = AutoTokenizer.from_pretrained(
        #    model_path,
        #    trust_remote_code=True,
        #    torch_dtype=torch.bfloat16
        #)

        llm_args = LLMArgs.builder() \
            .with_llm(llm) \
            .with_tokenizer(tokenizer) \
            .build()
        agent = LLMAgent(llm_args, env.action_space)

        agent_pop.append(
            # TODO: figure out how to do higher batch size
            PPO(agent=agent, index=i, batch_size=1)
        )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    mutations = create_mutation_obj(INIT_HP, MUTATION_PARAMS, device)    

    max_episodes = 50  # Max training episodes
    max_steps = 500  # Max steps per episode

    evo_epochs = 1#5  # Evolution frequency
    evo_loop = 1#3  # Number of evaluation episodes

    for idx_epi in trange(max_episodes):
        for agent in agent_pop:
            train_agent_one_episode(agent, env, max_steps)
    
        # Evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in agent_pop
            ]
    
            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f"Fitnesses: {[f'{fitness:.2f}' for fitness in fitnesses]}")
            print(
                f"100 fitness avgs: {[f'{np.mean(agent.fitness[-100:]):.2f}' for agent in agent_pop]}"
            )
    
            # Tournament selection and population mutation
            elite, agent_pop = tournament.select(agent_pop)
            agent_pop = mutations.mutation(agent_pop)
    
    env.close()
    
if __name__ == '__main__':
    main()