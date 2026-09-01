from typing import Callable

def replace_module(model, is_target_module:Callable, get_alternative:Callable) -> None:
    def __replace_module(model, full_name):
        for name, child in model.named_children():
            if is_target_module(full_name, child):
                new_module = get_alternative(full_name, child)
                setattr(model, name, new_module)
            __replace_module(child, full_name + '.' + name)
    __replace_module(model, '')