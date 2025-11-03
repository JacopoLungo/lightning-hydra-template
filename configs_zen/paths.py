from hydra_zen import make_config, store

PathsConfig = make_config(
    data_dir="data/",
    log_dir="logs/",
    output_dir="${hydra:runtime.output_dir}",
    work_dir="${hydra:runtime.cwd}",
)

store(PathsConfig, group="paths", name="default")
