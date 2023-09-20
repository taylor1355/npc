import json
import copy

class AgentArgs:
    def __init__(self, builder):
        self.field_names = []
        for field, _ in builder.specification.items():
            field_name = field[1:]
            field_val = getattr(builder, field)
            setattr(self, field_name, field_val)
            self.field_names.append(field_name)
    
    def asdict(self):
        field_dict = {}
        for field_name in self.field_names:
            field_dict[field_name] = getattr(self, field_name)
        return field_dict
    
    def __str__(self):
        return json.dumps(self.asdict(), indent=4)
        
    def builder():
        return AgentArgsBuilder()
        
class AgentArgsBuilder:
    REQUIRED = '__REQUIRED__'
    
    def create_field_setter(self, field):
        def field_setter(val):
            setattr(self, field, val)
            return self

        return field_setter
    
    def __init__(self):       
        self.specification = {         
            # Model architecture
            '_llm': AgentArgsBuilder.REQUIRED,
            '_tokenizer': AgentArgsBuilder.REQUIRED,

            # Model saving/loading
            # Optimizer config
            # Training logistics
            '_verbose': False,
        }

        for field, default in self.specification.items():
            default_val = None if default == AgentArgsBuilder.REQUIRED else default
            setattr(self, field, default_val)
            setattr(self, f'with{field}', self.create_field_setter(field))
            
    def clone(self):
        _clone = AgentArgsBuilder()
        for field in self.specification:
            setattr(_clone, field, copy.copy(getattr(self, field)))
        return _clone
          
    def build(self):
        for field, default in self.specification.items():
            if default == AgentArgsBuilder.REQUIRED and getattr(self, field) is None:
                raise ValueError(f'{field[1:]} is a required argument')
            
        return AgentArgs(self)