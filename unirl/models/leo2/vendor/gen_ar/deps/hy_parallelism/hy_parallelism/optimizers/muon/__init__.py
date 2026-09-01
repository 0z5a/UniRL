

def get_optimizer_factory_from_param_optimizer_mapping(param_optimizer_mapping):
    def factory(name, param):
        return param_optimizer_mapping.get(name, None)
    return factory