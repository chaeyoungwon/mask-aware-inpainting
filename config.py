class Config:
    # data
    data_root        = './data'
    image_size       = 128
    batch_size       = 16
    num_workers      = 0
    max_train_samples = 5000   # None이면 전체 사용

    # model — change conv_type to switch between models
    base_ch    = 64
    conv_type  = 'pconv'   # 'vanilla' | 'pconv' | 'gated'

    # train
    lr         = 2e-4
    num_epochs = 20
    save_dir   = './checkpoints'
    log_every  = 10

    # loss weights
    w_valid     = 1.0
    w_hole      = 6.0
    w_perceptual = 0.05
