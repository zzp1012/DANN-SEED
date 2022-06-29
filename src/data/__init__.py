# import internal libs
from utils import get_logger

def prepare_dataset(dataset: str) -> list:
    """prepare the dataset

    Args:
        dataset (str): the name of the dataset

    Return:
        train and valid set
    """
    logger = get_logger(__name__)
    logger.info(f"prepare the dataset {dataset}")
    if dataset == "SEED":
        import data.SEED as SEED
        domain_num, data = 5, []
        for d in range(domain_num):
            logger.info(f"leaving domain {d} out for validation")
            data.append(SEED.load(val_domain = d))
    else:
        raise NotImplementedError(f"the dataset {dataset} is not implemented.")
    return data