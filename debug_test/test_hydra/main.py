import dataclasses
import typing as tp
import hydra
from hydra.core.config_store import ConfigStore
import omegaconf as omgcf

@dataclasses.dataclass
class Config:
    agent: tp.Any
    crowd_sim: tp.Any 
    # misc
    seed: int = 1
    device: str = "cuda"

@hydra.main(config_path='.', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    print(cfg)


if __name__ == '__main__':
    main()
