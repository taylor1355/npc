from torch import nn
import torch

from .base_agent import BaseAgent

class Stream:
    def __init__(self, max_len):
        self.stream = []
        self.max_len = max_len
    
    def append(self, entry, timestamp):
        self.stream.append(f"<t={timestamp}> {entry} ")
        self.stream = self.stream[-self.max_len:]

    def __str__(self):
        return ''.join(self.stream)

class LLMHead(nn.Module):
    def __init__(self, llm, output_size, softmax_output=False):
        super().__init__()
        self.llm = llm
        self.head = nn.Linear(llm.config.hidden_size, output_size).to(llm.device)
        self.softmax_output = softmax_output

    def forward(self, input_ids, attention_mask=None):
        # TODO: 1) need to enable returning hidden states. 2) need to index into llm_output.hidden_states, as last_hidden_state does not exist
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        last_hidden_state = llm_output.hidden_states[-1]
        embedding = self.head(last_hidden_state[:, -1, :])
        if self.softmax_output:
            return torch.softmax(embedding, dim=-1)
        return embedding

class ActorNet(LLMHead):
    def __init__(self, llm, action_space):
        super().__init__(llm, len(action_space), softmax_output=True)

class CriticNet(LLMHead):
    def __init__(self, llm):
        super().__init__(llm, 1)

class LLMAgent(BaseAgent):
    def __init__(self, llm_args, action_space):
        self.llm = llm_args.llm
        print(self.llm.config)
        self.tokenizer = llm_args.tokenizer
        self.verbose = llm_args.verbose

        self.action_stream = Stream(5)
        self.thought_stream = Stream(5)
        self.observation_stream = Stream(1)

        self.num_internal_state_tokens = 128
        
        actor = ActorNet(self.llm, action_space)
        critic = CriticNet(self.llm)
        device = self.llm.device
        state_dim = (self.num_internal_state_tokens,)
        action_dim = len(action_space)
        super().__init__(actor, critic, device, state_dim, action_dim, discrete_actions=True)

    def prepare_state(self, internal_state):
        return internal_state

    # Updates the agent's thought and observation streams and returns its internal state 
    def update(self, state, reward, timestamp, thought_prompt_factory, state_to_str):
        self._update_observation_stream(state, reward, timestamp, state_to_str)
        self._update_thought_stream(thought_prompt_factory, timestamp)

        # potentially add a cls token to the end/beginning of internal_state
        internal_state = self.tokenizer(
            str(self.thought_stream),
            padding='max_length',
            truncation=True,
            max_length=self.num_internal_state_tokens,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return internal_state # TODO: also return attention mask

    def _update_thought_stream(self, thought_prompt_factory, timestamp):
        prompt = thought_prompt_factory(self.observation_stream, self.thought_stream)
        if self.verbose:
            print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # TODO: need to create streaming wrapper around llm, so that generation can stop on the <|im_end|> string
        output = self.llm.generate(input_ids, temperature=0.5, repetition_penalty=1.25, do_sample=True, max_new_tokens=50)
        thought = self.tokenizer.decode(output[0][input_ids.shape[-1]:])
        print(thought)
        self.thought_stream.append(thought, timestamp)

    # Limited to most recent observation for now
    def _update_observation_stream(self, state, reward, timestamp, state_to_str):
        state_str = state_to_str(state)
        self.observation_stream.append(f'State=[{state_str}], Reward={reward}', timestamp)

    # TODO: override clone