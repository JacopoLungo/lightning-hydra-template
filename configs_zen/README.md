# Hydra-Zen Migration Example

This directory demonstrates how to migrate from YAML configs to typed Python configs using [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen).

## What's Included

- `model.py` - Typed configs for models, optimizers, schedulers
- `data.py` - Typed configs for data modules
- `__init__.py` - Config registration with Hydra's ConfigStore
- `../train_zen.py` - Example training script using typed configs

## Side-by-Side Comparison

### YAML Config (configs/model/mnist.yaml)
```yaml
_target_: src.models.mnist_module.MNISTLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 10

net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 64
  lin2_size: 128
  lin3_size: 64
  output_size: 10

compile: false
```

### Hydra-Zen Config (configs_zen/model.py)
```python
from hydra_zen import builds
import torch
from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.mnist_module import MNISTLitModule

# Build network config
SimpleDenseNetConfig = builds(
    SimpleDenseNet,
    input_size=784,
    lin1_size=64,
    lin2_size=128,
    lin3_size=64,
    output_size=10,
)

# Build optimizer config
AdamConfig = builds(
    torch.optim.Adam,
    lr=0.001,
    weight_decay=0.0,
    populate_full_signature=True,
    zen_partial=True,  # Equivalent to _partial_: true
)

# Build scheduler config
ReduceLROnPlateauConfig = builds(
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    mode="min",
    factor=0.1,
    patience=10,
    populate_full_signature=True,
    zen_partial=True,
)

# Compose full model config
MNISTModelConfig = builds(
    MNISTLitModule,
    net=SimpleDenseNetConfig,
    optimizer=AdamConfig,
    scheduler=ReduceLROnPlateauConfig,
    compile=False,
    populate_full_signature=True,
)
```

## Key Benefits

### 1. Type Safety
```python
# YAML: No type checking, typos cause runtime errors
optimizer:
  lr: 0.001
  weight_decoy: 0.0  # Typo! Only caught at runtime

# Hydra-Zen: IDE catches typos immediately
AdamConfig = builds(
    torch.optim.Adam,
    lr=0.001,
    weight_decoy=0.0,  # IDE shows error: no parameter 'weight_decoy'
)
```

### 2. IDE Support
- Full autocomplete for all parameters
- Jump to definition (Cmd/Ctrl + Click)
- Find all references
- Automatic refactoring support

### 3. Programmatic Generation
```python
# Create configs dynamically with Python logic
def make_data_config(batch_size: int, num_workers: int = 0):
    return builds(
        MNISTDataModule,
        data_dir="${paths.data_dir}",
        batch_size=batch_size,
        num_workers=num_workers,
        populate_full_signature=True,
    )

# Generate multiple configs easily
configs = [make_data_config(bs) for bs in [32, 64, 128, 256]]
```

### 4. Easier Testing
```python
# Test configs directly in Python
def test_model_config():
    config = MNISTModelConfig()
    assert config.optimizer.lr == 0.001
    assert config.compile is False
    # Full type checking and validation!
```

## Usage Examples

### Basic Training
```bash
# Use default hydra-zen configs
python src/train_zen.py

# This uses the registered configs from configs_zen/__init__.py
```

### Override Parameters
```bash
# Override just like YAML configs
python src/train_zen.py model.optimizer.lr=0.01 data.batch_size=256
```

### Use Different Variants
```bash
# Use pre-registered variant configs
python src/train_zen.py model=mnist_large_lr data=mnist_prod
```

### Hybrid Approach (Recommended for Migration)
```bash
# Mix hydra-zen configs with YAML configs
python src/train_zen.py \
    model=mnist \          # From hydra-zen
    data=mnist_prod \      # From hydra-zen
    trainer=gpu \          # From YAML
    logger=wandb \         # From YAML
    callbacks=default      # From YAML
```

## Migration Strategy

### Phase 1: Gradual Migration (Recommended)
1. Keep existing YAML configs for trainer, callbacks, loggers
2. Migrate model and data configs to hydra-zen
3. Use hybrid approach with both systems
4. Team gets familiar with hydra-zen

### Phase 2: Expand Coverage
1. Migrate callbacks to hydra-zen
2. Migrate loggers to hydra-zen
3. Create typed experiment configs
4. Most configs now in Python

### Phase 3: Complete Migration (Optional)
1. Move remaining configs to Python
2. Remove YAML configs directory
3. Fully typed configuration system

## Comparison Matrix

| Feature | YAML Configs | Hydra-Zen |
|---------|-------------|-----------|
| Type Safety | ❌ Runtime only | ✅ Compile-time |
| IDE Autocomplete | ❌ No | ✅ Full support |
| Refactoring | ❌ Manual | ✅ Automatic |
| Validation | ❌ Runtime errors | ✅ Early validation |
| Version Control | ✅ Easy to read | ⚠️ More verbose |
| Non-programmer friendly | ✅ Simple YAML | ❌ Requires Python |
| Programmatic generation | ❌ Not possible | ✅ Full Python power |
| Learning curve | ✅ Minimal | ⚠️ Steeper |

## Installation

Add hydra-zen to your dependencies:
```bash
pip install hydra-zen

# Or add to requirements.txt
echo "hydra-zen>=0.12.0" >> requirements.txt
```

## Next Steps

1. **Try the example**: Run `python src/train_zen.py` to see it in action
2. **Experiment with overrides**: Try different parameter combinations
3. **Create your own configs**: Add new model or data variants
4. **Decide on migration strategy**: Full, hybrid, or keep YAML

## Resources

- [Hydra-Zen Documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Migration Guide](https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/add_ddp.html)
