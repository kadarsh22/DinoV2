import yaml
import logging

def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def setup_logging(rank):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    return logger

def unwrap_fsdp(m):
    return getattr(m, "_fsdp_wrapped_module", m)