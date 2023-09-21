import os
import tarfile
import shutil
import tempfile
import time
import pickle
import matplotlib.pyplot as plt

from src import config
import argparse
from types import SimpleNamespace
import wandb

from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch
import torch.nn.parallel
import torch.distributed as dist
import logging

logging.basicConfig(level=logging.INFO)

logging.info(print_config())

# set seed to reproducibility
set_determinism(seed=0)


# Convert labels to multi channels
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is necrosis
    label 2 is edema
    label 3 is activo
    The possible classes are N (necrosis), E (edema)
    and TA (active).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 1 necro
            result.append(d[key] == 1)
            # label 2 is ET
            result.append(d[key] == 2)
            # merge labels 3, 4 and 3 to construct activo
            result.append(torch.logical_or(d[key] == 3, d[key] == 4))

            d[key] = torch.stack(result, axis=0).float()
        return d


# Do transformations to create tensors
# Transformations for Data train
train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=[224, 224, 144], random_size=False
        ),  # Random transformation for
        RandFlipd(
            keys=["image", "label"], prob=0.5, spatial_axis=0
        ),  # data augmentation
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(
            keys="image", nonzero=True, channel_wise=True
        ),  # Zscore normalization for intensity
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)

# Transformations for Data validation
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


# here we don't cache any data in case out of memory issue
def main():
    # create args parser
    parser = argparse.ArgumentParser(description="Pipeline para entrenar SegResNet")
    parser.add_argument(
        "--download",
        default=False,
        action="store_true",
        help="Download Dataset",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs",
    )

    args = parser.parse_args()

    train_ds = DecathlonDataset(
        root_dir=config.DATASET,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=args.download,
        cache_rate=0.0,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_ds = DecathlonDataset(
        root_dir=config.DATASET,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=args.download,
        cache_rate=0.0,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Select device gpu or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Cargar la clave API desde una variable de entorno
    logging.info("Logging in WandB")
    api_key = os.environ.get("WANDB_API_KEY")
    # Iniciar sesión en W&B
    wandb.login(key=api_key)

    # DATA_DIR = Path('./data/')
    SAVE_DIR = config.DATASET
    # SAVE_DIR.mkdir(exist_ok=True, parents=True)
    DEVICE = device

    # HIPER PARAMETER CONFIGURATION
    config_train = SimpleNamespace(
        # network hyperparameters
        init_filters=16,
        dropout_prob=0.2,
        # training hyperparameters
        max_epochs=args.epochs,
        lrate=1e-4,
        weight_decay=1e-5,
        batch_size=1,
        # Post
        threshold=0.5,
        # Train type
        use_scaler=True,
        use_autocast=True,
    )

    # define inference method
    def inference(input, model, VAL_AMP=config_train.use_autocast):
        # if config_train.use_autocast:
        #    VAL_AMP = True
        # else:
        #    VAL_AMP = False

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    # create a wandb run
    run = wandb.init(
        project="SegResNet_Cerebrum", job_type="train", config=config_train
    )

    # we pass the config back from W&B
    config_train = wandb.config

    max_epochs = config_train.max_epochs
    val_interval = 1

    logging.info("Creating Model")
    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=config_train.init_filters,
        in_channels=4,
        out_channels=3,
        dropout_prob=config_train.dropout_prob,
    )

    # if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    # model = nn.DataParallel(model)

    model.to(device)

    loss_function = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(), config_train.lrate, weight_decay=config_train.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config_train.max_epochs
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=config_train.threshold)]
    )

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    logging.info("Training Model")
    best_metric = 1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    # last_epoch = 24

    total_start = time.time()
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        epoch_start = time.time()
        logging.info("-" * 10)
        logging.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()

            if config_train.use_scaler:
                # with autocast
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # without autocast
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            logging.info(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
            # wandb
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
            torch.cuda.empty_cache()
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch_loss < best_metric:
            best_metric = epoch_loss
            best_metric_epoch = epoch + 1

            best_model_file = os.path.join(SAVE_DIR, "best_metric_model.pth")

            torch.save(
                model.state_dict(),
                best_model_file,
            )
            torch.cuda.empty_cache()
            logging.info("saved new best metric model")

        logging.info(
            f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}"
        )
    # wandb save model
    artifact_name = f"{wandb.run.id}_best_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(best_model_file)
    wandb.log_artifact(at, aliases=[f"epoch_{epoch}"])

    total_time = time.time() - total_start
    logging.info(f"Total time: {total_time}")

    # finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
