import os
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
jax.config.update("jax_enable_x64", True) 
import equinox as eqx
import sys
sys.path.append('..')
sys.path.append('../..')

import hydra
import logging
from omegaconf import DictConfig

from onsagernet.trainers import MLETrainer
from utils_Kaczmarz import load_data, build_OnsagerNetHD2, generate_data
 
from logging import Logger

from utils_Kaczmarz import get_filter_spec 

def train_model(config: DictConfig, logger: Logger= None, lr=0.01, batch_size=1, path: str ="test", model_seed=0) -> None:
    """Main training routine

    Args:
        config (DictConfig): configuration object
    """
    subruntime_dir = os.path.join(path, f"seed{model_seed}_models")
    os.makedirs(subruntime_dir, exist_ok=True)
    N=500000
    print(subruntime_dir)
    logger.info(f"Device {jax.devices()}...")

    cache_dir = "OriginalData/relat4_{}/batch_size_{}".format(lr, batch_size)
    if config.data.generate:
        logger.info(f"Generating data...")
        train_dataset = generate_data(total_seed=int(N*0.8), batch_size=batch_size, lr=lr)
        test_dataset = generate_data(total_seed=int(N*0.2), batch_size=batch_size, lr=lr)
        logger.info(f"Data generated...")
        logger.info("Caching dataset...")
        train_dataset.save_to_disk(os.path.join(cache_dir, "train"))
        test_dataset.save_to_disk(os.path.join(cache_dir, "test"))


    logger.info(f"Loading dataset ...")
    train_dataset, test_dataset = load_data(cache_dir)

    logger.info(f"Building HD2 ...") 
    model = build_OnsagerNetHD2(config, seed=model_seed)
    filter_spec = get_filter_spec(model)
    
    trainer = MLETrainer(opt_options=config.train.opt, rop_options=config.train.rop)

    logger.info(f"Training {config.Model_name} for {config.train.num_epochs} epochs...")
    trained_model, _, _ = trainer.train(
        model=model,
        dataset=train_dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        test_dataset= test_dataset,
        logger=logger,
        filter_spec=filter_spec,
        checkpoint_dir=subruntime_dir,
        checkpoint_every=config.train.checkpoint_every,
        print_every=config.train.print_every,
    )
    
    eqx.tree_serialise_leaves(os.path.join(subruntime_dir, "model.eqx"), trained_model)
    

@hydra.main(config_path="./config", config_name="main_Kaczmarz_res_scale", version_base=None)
def main(config: DictConfig) -> None:
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__)
    for seed in [10,20,30,40]:
        train_model(config, logger, config.data.lr, config.data.batch_size, runtime_dir, model_seed=seed)
    logger.info(f"Saving output to {runtime_dir}")

if __name__ == "__main__":
    main()
