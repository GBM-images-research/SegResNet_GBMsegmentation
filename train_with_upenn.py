# Loading functions
import os
import tarfile
import shutil
import tempfile
import time
import pickle
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset, TciaDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset
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
    CropForegroundd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ResampleToMatchd,
    CropForegroundd,
)
from monai.utils import set_determinism
import torch
import torch.nn.parallel
import torch.distributed as dist

from src.get_data import CustomDataset
import numpy as np
from scipy import ndimage
from types import SimpleNamespace
import wandb
import logging

logging.basicConfig(level=logging.INFO)


# Funciones personalizadas


def fill_holes_3d(mask):
    # Rellenar huecos en la máscara 3D
    filled_mask = ndimage.binary_fill_holes(mask)
    return filled_mask


def expand_mask_3d_td(
    mask, edema, distance_cm_max=0.5, distance_cm_min=0.1, voxel_size=0.1
):
    distance_pixels_max = int(distance_cm_max / voxel_size)
    distance_pixel_min = int(distance_cm_min / voxel_size)

    # Calcular la transformada de distancia
    distance_transform = ndimage.distance_transform_edt(np.logical_not(mask))

    # Crear la nueva máscara alrededor del tumor core
    # expanded_mask_distance = distance_transform >= distance_threshold
    expanded_mask = np.logical_and(
        distance_transform >= distance_pixel_min,
        distance_transform <= distance_pixels_max,
    )

    # Restar la máscara original para obtener solo la región expandida
    exterior_mask = np.logical_and(expanded_mask, np.logical_not(mask))
    # Hacer un AND con el edema para eliminar zonas externas a este
    exterior_mask = np.logical_and(exterior_mask, edema)

    return torch.from_numpy(exterior_mask)


class ConvertToMultiChannel_with_infiltration(MapTransform):
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
            necro = d[key] == 1
            # result.append(necro)

            # label 2 is Edema
            edema = d[key] == 2
            # result.append(edema)

            # merge labels 3, 4 and 3 to construct activo
            active = torch.logical_or(d[key] == 3, d[key] == 4)
            # result.append(active)

            # Determinar las ROI cercana y lejana al Tumor Core
            tumor_core_mask = np.logical_or(necro, active)

            # Rellenar los huecos en la máscara
            filled_tumor_core = fill_holes_3d(tumor_core_mask)
            # result.append(torch.from_numpy(filled_tumor_core))

            # Definir el tamaño de voxel en centímetros (ajusta según tus datos)
            voxel_size_cm = 0.1

            # Expandir la máscara de 1 cm alrededor del tumor core (N_ROI)
            N_roi = expand_mask_3d_td(
                filled_tumor_core,
                edema=edema,
                distance_cm_max=0.5,
                distance_cm_min=0.1,
                voxel_size=voxel_size_cm,
            )
            result.append(N_roi)

            F_roi = expand_mask_3d_td(
                filled_tumor_core,
                edema=edema,
                distance_cm_max=10,
                distance_cm_min=1,
                voxel_size=voxel_size_cm,
            )
            result.append(F_roi)
            result.append(edema)

            d[key] = torch.stack(result, axis=0).float()
        return d


# Transformaciones
t_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannel_with_infiltration(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"], source_key="label", margin=[112, 112, 72]
        ),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=[112, 112, 72], random_size=False
        ),  # [224, 224, 144]
    ]
)

v_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannel_with_infiltration(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)


# Creando el modelo
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# DATA_DIR = Path('./data/')
SAVE_DIR = "./Dataset"
# SAVE_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = device

#################################
# HIPER PARAMETER CONFIGURATION #
#################################
config_train = SimpleNamespace(
    # network hyperparameters
    init_filters=16,
    dropout_prob=0.2,
    # training hyperparameters
    max_epochs=100,
    lrate=1e-4,
    weight_decay=1e-5,
    batch_size=1,
    # Post
    threshold=0.5,
    # Train type
    use_scaler=True,
    use_autocast=True,
    GT="nroi + froi + edema",
)


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


# Cargar la clave API desde una variable de entorno
logging.info("Logging in WandB")
api_key = os.environ.get("WANDB_API_KEY")
# Iniciar sesión en W&B
wandb.login(key=api_key)


####################################
# Load DATASET and training modelo #
####################################
def main(config_train):
    dataset_path = "./Dataset/Dataset_30_casos/"

    train_set = CustomDataset(
        dataset_path, section="train", transform=t_transform
    )  # t_transform
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)

    im_t = train_set[0]
    # (im_t["image"].shape)
    print(im_t["label"].shape)

    val_set = CustomDataset(
        dataset_path, section="valid", transform=v_transform
    )  # v_transform
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # im_v = val_set[0]
    # print(im_v["image"].shape)
    # print(im_v["label"].shape)

    # create a wandb run
    run = wandb.init(project="SegResNet_UPENN", job_type="train", config=config_train)

    # we pass the config back from W&B
    config_train = wandb.config

    # Create SegResNet, DiceLoss and Adam optimizer
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],  # 4
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=11,
        out_channels=3,
        dropout_prob=config_train.dropout_prob,
    )

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

    ##########################################################
    # Comenzar entrenamiento
    ##########################################################
    val_interval = 1

    max_epochs = config_train.max_epochs

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_nroi = []  # tc nroi
    metric_values_froi = []  # wt
    metric_values_et = []

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
                f"{step}/{len(train_set) // train_loader.batch_size}"
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

        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        wandb.log(
            {
                "Epoch_Average_Loss": epoch_loss,
            }
        )

        # Evaluar en validación
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = inference(val_inputs, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_nroi = metric_batch[0].item()
                metric_values_nroi.append(metric_nroi)
                metric_froi = metric_batch[1].item()
                metric_values_froi.append(metric_froi)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                wandb.log(
                    {
                        "Nroi_dice": metric_nroi,
                        "Froi_dice": metric_froi,
                        "Edema_dice": metric_et,
                    }
                )
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(SAVE_DIR, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" Nroi: {metric_nroi:.4f} Froi: {metric_froi:.4f}  Edema: {metric_et:.4f}"
                    f"\n best mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(
            f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}"
        )
    total_time = time.time() - total_start
    # wandb save model
    artifact_name = f"{wandb.run.id}_best_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(os.path.join(SAVE_DIR, "best_metric_model.pth"))
    wandb.log_artifact(at, aliases=[f"epoch_{epoch}"])

    total_time = time.time() - total_start
    logging.info(f"Total time: {total_time}")

    # finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main(config_train)
