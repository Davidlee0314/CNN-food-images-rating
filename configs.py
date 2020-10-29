class configs:
    # image size for training 
    img_size = (224, 224)

    # traing parameters
    random_seed = 7
    batch_size = 32
    lr = 1e-3
    epoch_mean = 20
    epoch_std = 20

    # file directory
    train_label_dir = './train/label_train.txt'
    train_dir = './train'
    val_dir = './validation'

    # store training device
    device = None

    # use pretrained models
    use_pretrain = False
