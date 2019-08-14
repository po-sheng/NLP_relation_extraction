class Token:
    def __init__(self, id: int, text: str, pos, ner: str=None, dep_type: str=None, dep_head: str=None):
    
class Relation:
    def __init__(self, type: str=str, arg1: Entity, arg2: Entity):
        
class Entity:
    def __init__(self, id: int, tokens: List[Token], type: str=None, SO: str=None):
    
class Sentence:
    def __init__(self, tokens: List[Token], string: str=None):
        
class Document:
    def __init__(self, sentences: List[Sentence]):