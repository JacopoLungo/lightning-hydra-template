import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

from hydra_zen import store, builds, instantiate, ZenStore

# --- 1. Define your Python-based Configs ---
# We'll configure this simple class
@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8080

# --- 2. Create and Register Configs with hydra-zen ---

# We'll create a ZenStore to manage our configs
# Note: deferred_hydra_store=False registers configs immediately
#       with Hydra's global ConfigStore.
#       This is necessary so @hydra.main can find them.
store = ZenStore(deferred_hydra_store=False)

# Create a config named "config" using builds(),
# which will configure our ServerConfig
AppConfig = builds(ServerConfig, pots=2000, populate_full_signature=True)
store(AppConfig, name="config")


# --- 3. Use the @hydra.main decorator as usual ---
#
# This function will receive a DictConfig, NOT an instantiated object,
# because @hydra.main does not know about hydra-zen's 
# automatic instantiation.
@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("--- Inside main ---")
    print(type(cfg._target_))
    print(f"Received DictConfig:\n{OmegaConf.to_yaml(cfg)}")
    
    # You can still use hydra-zen to instantiate the config *manually*
    # This gives you back the Python object
    server: ServerConfig = instantiate(cfg)
    
    print("\n--- Instantiated Config ---")
    print(f"Type: {type(server)}")
    print(f"Server running at: {server.host}:{server.port}")


# --- 4. Run the decorated function ---
if __name__ == "__main__":
    # When this script is run, @hydra.main takes over
    # It will find the "config" we registered in the
    # global ConfigStore using hydra-zen.
    main()