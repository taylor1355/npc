import json
import copy

class LLMArgs:
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
        return LLMArgsBuilder()
        
class LLMArgsBuilder:
    REQUIRED = '__REQUIRED__'
    
    def create_field_setter(self, field):
        def field_setter(val):
            setattr(self, field, val)
            return self

        return field_setter
    
    def __init__(self):       
        self.specification = {         
            '_llm': LLMArgsBuilder.REQUIRED,
            '_tokenizer': LLMArgsBuilder.REQUIRED,
            '_verbose': False,
        }

        for field, default in self.specification.items():
            default_val = None if default == LLMArgsBuilder.REQUIRED else default
            setattr(self, field, default_val)
            setattr(self, f'with{field}', self.create_field_setter(field))
            
    def clone(self):
        _clone = LLMArgsBuilder()
        for field in self.specification:
            setattr(_clone, field, copy.copy(getattr(self, field)))
        return _clone
          
    def build(self):
        for field, default in self.specification.items():
            if default == LLMArgsBuilder.REQUIRED and getattr(self, field) is None:
                raise ValueError(f'{field[1:]} is a required argument')
            
        return LLMArgs(self)