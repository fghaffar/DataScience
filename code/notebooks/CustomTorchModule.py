from typing import Any, TypeVar, Iterable, Tuple, Dict
import torch.nn as nn

class Module(nn.Module):
    def __init__subclass__(cls, **kwargs):
        if cls.__dict__.get('__call__', None) is None:
            # raise TypeError('Module subclasses must override __call__ method')
            return
        
        setattr(cls, 'forward', cls.__dict__['__call__'])
        delattr(cls, '__call__')
    
    @property
    def device(self):
        
