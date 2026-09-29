from fractaltokenizer_v2.imagenet_training import TrainingConfig, run_comparison


config = TrainingConfig(
    variants=("fractal_128", "vit_rect_128", "vit_patch16_256"),
    seeds=(42,),
    stage_to_shared_memory=False,
    output_root="runs/imagenet100_smoke",
    epochs=1,
    micro_batch_size=8,
    gradient_accumulation_steps=1,
    workers=4,
    model_dim=96,
    depth=2,
    num_heads=4,
    mlp_ratio=2.0,
    mixup_alpha=0.2,
    compile_model=False,
    resume=False,
    telegram_enabled=False,
    telegram_plot_every=1,
    max_train_batches=2,
    max_val_batches=2,
)

run_comparison(config)
