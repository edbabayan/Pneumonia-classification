from pytorch_lightning.callbacks import ModelCheckpoint


checkpoint_callback = ModelCheckpoint(
    monitor='Val accuracy',
    save_top_k=10,
    mode='max'
)
