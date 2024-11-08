import json
import logging

from trainer import Trainer
from utils.logging import LogFile, create_logger

##########################################################################################

with open("./config/trainer.json", "r") as file:
    config = json.load(file)
    env_params = config["env_params"]
    model_params = config["model_params"]
    optimizer_params = config["optimizer_params"]
    trainer_params = config["trainer_params"]

trainer_params["use_cuda"] = not trainer_params["debug_mode"]

logger_params = {"log_file": LogFile(desc="train")}

##########################################################################################
# main


def main():

    trainer_params["train_episodes"] = (
        trainer_params["number_of_batches"] * trainer_params["train_batch_size"]
    )

    if trainer_params["debug_mode"]:
        trainer_params["epochs"] = 2
        trainer_params["train_episodes"] = 10
        trainer_params["train_batch_size"] = 4

    else:
        trainer_params["train_episodes"] = (
            trainer_params["number_of_batches"] * trainer_params["train_batch_size"]
        )

    # if os.path.exists('./logs/log_train.txt'):
    #    os.remove('./logs/log_train.txt')

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    # copy_all_src(trainer.result_folder)

    trainer.run()


def _print_config():
    logger = logging.getLogger("root")
    logger.info(f"DEBUG_MODE: {trainer_params['debug_mode']}")
    logger.info(
        f"USE_CUDA: {trainer_params['use_cuda']}, CUDA_DEVICE_NUM: {trainer_params['cuda_device_num']}"
    )

    [
        logger.info(f"{g_key}{globals()[g_key]}")
        for g_key in globals()
        if g_key.endswith("params")
    ]


##########################################################################################
if __name__ == "__main__":
    main()
