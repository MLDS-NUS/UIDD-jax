import os
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.48"

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import sys
sys.path.append('..')
sys.path.append('../..')
from utils.data import shrink_trajectory_len

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from onsagernet.trainers import MLETrainer
from utils_reduced_polymer import load_data, build_OnsagerNet, build_OnsagerNetHD2_scale
 
from logging import Logger

from utils_reduced_polymer import get_filter_spec 

def train_model(config: DictConfig, logger: Logger= None, dataset_name: str = "F23_10_T1", path: str ="test", scale:float=1.0) -> None:
    """Main training routine

    Args:
        config (DictConfig): configuration object
    """
    subruntime_dir = os.path.join(path, dataset_name)
    os.makedirs(subruntime_dir, exist_ok=True)

    print(subruntime_dir)

    logger.info(f"Loading dataset {dataset_name}...")
    train_dataset, test_dataset = load_data(dataset_name)
    print(train_dataset)
    train_traj_len = config.train.get("train_traj_len", None)
    if train_traj_len is not None:
        train_dataset = shrink_trajectory_len(
            train_dataset, train_traj_len
        )  # change the trajectory length to improve GPU usage
        test_dataset = shrink_trajectory_len(
            test_dataset, train_traj_len
        )  

    logger.info(f"Building {config.Model_name} ...")
    if config.Model_name == 'Onsager':
        model = build_OnsagerNet(config)
        filter_spec=None
    elif config.Model_name == 'HD2':
        model = build_OnsagerNetHD2_scale(config, scale)
        filter_spec = get_filter_spec(model)
    else:
        raise ValueError("wrong model")
    
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
    

@hydra.main(config_path="./config", config_name="main_reduced_polymer", version_base=None)
def main(config: DictConfig) -> None:
    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = logging.getLogger(__name__) 
    data_set_key_list=["F23_10_T1", "F16_20_T1", "F11_60_T1", 
                    #    "F4_63_T1", "F3_93_T1", "F2_78_T1", "F2_31_T1", "F1_62_T1"
                  ]
    def scale_number(F):
        return 2**(F/3)/3
    scale_list={"F23_10_T1": scale_number(23.1), "F16_20_T1": scale_number(16.2), "F11_60_T1":scale_number(11.6),"F4_63_T1": scale_number(4.63), 
                "F3_93_T1": scale_number(3.93), "F2_78_T1": scale_number(2.78), "F2_31_T1": scale_number(2.31), "F1_62_T1": scale_number(1.62)}

    for data_set_key in data_set_key_list:
        train_model(config, logger, data_set_key, runtime_dir, scale_list[data_set_key])
    logger.info(f"Saving output to {runtime_dir}")

if __name__ == "__main__":
    main()
