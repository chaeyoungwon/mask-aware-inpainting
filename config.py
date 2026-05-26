class Config:
    # data
    data_root         = './data'
    image_size        = 128
    batch_size        = 32
    num_workers       = 2
    max_train_samples = 5000  # None이면 전체 사용

    # model — change conv_type to switch between models
    base_ch   = 32
    conv_type = 'gated'   # 'vanilla' | 'pconv' | 'gated'

    # reproducibility
    seed = 42

    # train
    lr         = 2e-4
    num_epochs = 30
    save_dir   = './checkpoints'
    log_every  = 20

    # scheduler — CosineAnnealingLR
    use_scheduler = True
    T_max         = 30   # num_epochs와 동일하게
    eta_min       = 1e-6

    # early stopping
    early_stopping_patience = 5 

    # loss weights
    w_valid      = 1.0
    w_hole       = 6.0
    w_perceptual = 0.05
