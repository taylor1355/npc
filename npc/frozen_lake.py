from .llm_args import LLMArgs
from .llm_agent import LLMAgent
from .prompt_util import create_prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import gymnasium as gym
import torch

# TODO: this should be a subclass of BaseEnv or maybe the FrozenLakeEnv class
class FrozenLake:
    def __init__(self, is_slippery=False):
        self.is_slippery = is_slippery
        self.env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=is_slippery)
        self.timestamp = 0
        self.done = False

        self.env_description = ''.join([
            'I am OrcaPhi. The following is my internal dialogue as an intelligent AI agent.',
            'I am playing a video game where I must avoid walking into any holes while',
            ' traveling over a frozen lake to find the goal tile. I can safely move onto floor tiles but cannot move through walls. I ',
            ' am able to see both the goal and any holes from one tile away. The faster I reach the goal the better.'
        ])
        if self.is_slippery:
            self.env_description += ' Due to the slippery nature of the frozen lake, I may accidentally move perpendicular to the direction I intend to move sometimes '

        self.action_space = {
            0: 'West',
            1: 'South',
            2: 'East',
            3: 'North'
        }
        

    def thought_prompt_factory(self, observation_stream, thought_stream):
        system_prompt = create_prompt('system', '\n'.join([
            self.env_description,
            f'My observations: {observation_stream}',
            f'My past thoughts: {thought_stream}',
            f'Possible Actions: {self.action_space}'
        ]))
        return '\n'.join([
            system_prompt,
            create_prompt('user', 'Choose which action you will take.'),
            create_prompt('assistant', '', terminate=False)
        ])

    def action_prompt_factory(self, thought_stream):
        system_prompt = create_prompt('system', '\n'.join([
            self.env_description,
            f'My plans: {thought_stream}',
            f'Possible Actions: {self.action_space}'
        ]))
        return '\n'.join([
            system_prompt,
            create_prompt('user', 'What action are you going to take?'),
            create_prompt('assistant', 'I will take action "', terminate=False)
        ])

    def describe_tile(self, row, col):
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return 'wall'
        else:
             env_letter = self.env.desc[row, col].decode('utf-8')
             match env_letter:
                 case 'S':
                     return 'start'
                 case 'F':
                     return 'floor'
                 case 'H':
                     return 'hole'
                 case _:
                     return 'unknown'

    def state_to_str(self, state):
        row, col = state // 4, state % 4
        surroundings = [
            f"North: {self.describe_tile(row - 1, col)}",
            f"South: {self.describe_tile(row + 1, col)}",
            f"West: {self.describe_tile(row, col - 1)}",
            f"East: {self.describe_tile(row, col + 1)}",
        ]
        return f"Row {row}, Col {col}, {', '.join(surroundings)}"

    def reset(self):
        self.timestamp = 0
        self.done = False
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    def test(self, agent=None):
        if agent is None:
            model_path = "princeton-nlp/Sheared-LLaMA-1.3B"#Open-Orca/oo-phi-1_5"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
            )
            tokenizer.pad_token = tokenizer.eos_token

            llm_args = LLMArgs.builder() \
                .with_llm(llm) \
                .with_tokenizer(tokenizer) \
                .build()
            agent = LLMAgent(llm_args, self.action_space)

        state, _ = self.reset()
        reward = 0

        while not self.done:
            input("Press Enter to take a step...")
            print('\n\n' + '=' * 80)

            internal_state = agent.update(
                state, reward, self.timestamp,
                self.thought_prompt_factory,
                self.state_to_str
            )
            action, action_logprob, dist_entropy, state_values = self.agent.getAction(internal_state)
            action = action.item()
            print(action)
            state, reward, self.done, _, _ = self.step(action)
            self.env.render()

            self.timestamp += 1

            print(f'| t={self.timestamp}')
            print(f'| Observation Stream: [{self.agent.observation_stream}]')
            print(f'| Thought Stream: [{self.agent.thought_stream}]')
            print(f'| Action Stream: [{self.agent.action_stream}]')

if __name__ == '__main__':
    env = FrozenLake()
    env.test()