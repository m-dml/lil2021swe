import hydra
from src.train import train
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg : DictConfig) -> None:
    
    train(cfg)
        
if __name__ == '__main__':
    main()