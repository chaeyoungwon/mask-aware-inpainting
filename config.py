class Config:
    # data
    data_root        = './data'
    image_size       = 256
    batch_size       = 16
    num_workers      = 4
    max_train_samples = 20000  # None이면 전체 사용

    # model
    base_ch    = 64
    conv_type  = 'vanilla'   # 'vanilla' | 'pconv' | 'gated'

    # train
    lr         = 2e-4
    num_epochs = 100
    save_dir   = './checkpoints'
    log_every  = 50

    # loss weights
    w_valid     = 1.0
    w_hole      = 6.0
    w_perceptual = 0.05
