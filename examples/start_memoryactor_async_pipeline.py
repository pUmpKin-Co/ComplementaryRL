import argparse
import os
from enum import Enum

import yaml
from dacite import Config, from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.multi_agentic_config import MemoryActorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    ppo_config = from_dict(data_class=MemoryActorConfig, data=resolved_config, config=Config(cast=[Enum]))

    try:
        out_dir = ppo_config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        config_output_dir = os.path.join(out_dir, "config.yaml")
        with open(config_output_dir, "w") as file_obj:
            yaml.dump(resolved_config, file_obj, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        print(f"[WARN] Failed to save resolved config: {exc}")

    init()
    from roll.pipeline.agentic.multi_agentic_pipeline import FullyAsyncMemoryActorPipeline

    pipeline = FullyAsyncMemoryActorPipeline(pipeline_config=ppo_config)
    pipeline.run()


if __name__ == "__main__":
    main()
