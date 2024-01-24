# Loading functions
import os
import time
from monai.data import DataLoader, decollate_batch

# from monai.losses import DiceLoss
# from monai.inferers import sliding_window_inference
# from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet

from monai.transforms import (
    #     Activations,
    #     AsDiscrete,
    #     Compose,
    #     LoadImaged,
    MapTransform,
    #     NormalizeIntensityd,
    #     Orientationd,
    #     RandSpatialCropd,
    #     CropForegroundd,
    #     Spacingd,
    #     EnsureTyped,
    #     EnsureChannelFirstd,
    #     CropForegroundd,
)

import torch
import torch.nn.parallel

from src.get_data import CustomDataset
import numpy as np
from scipy import ndimage
from types import SimpleNamespace
import wandb
import logging

#####
import json
import shutil
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data

# from monai.data import decollate_batch
from functools import partial

import torch

####

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
            result.append(edema)  # comentar para eliminar edema de GT

            d[key] = torch.stack(result, axis=0).float()
        return d


# # Transformaciones
# t_transform = Compose(
#     [
#         LoadImaged(keys=["image", "label"], allow_missing_keys=True),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannel_with_infiltration(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         CropForegroundd(
#             keys=["image", "label"], source_key="label", margin=[112, 112, 72]
#         ),
#         RandSpatialCropd(
#             keys=["image", "label"], roi_size=[112, 112, 72], random_size=False
#         ),  # [224, 224, 144]
#     ]
# )


# v_transform = Compose(
#     [
#         LoadImaged(keys=["image", "label"], allow_missing_keys=True),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannel_with_infiltration(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#     ]
# )


# # Creando el modelo
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # DATA_DIR = Path('./data/')
# SAVE_DIR = "./Dataset"
# # SAVE_DIR.mkdir(exist_ok=True, parents=True)
# DEVICE = device

#################################
# HIPER PARAMETER CONFIGURATION #
#################################

### Hyperparameter
roi = (128, 128, 128)  # (128, 128, 128)
batch_size = 1
sw_batch_size = 2
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 1
# train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

config_train = SimpleNamespace(
    roi=roi,
    batch_size=batch_size,
    sw_batch_size=sw_batch_size,
    fold=fold,
    infer_overlap=infer_overlap,
    max_epochs=max_epochs,
    val_every=val_every,
    GT="nroi + froi + edema",  # modifica para eliminar edema
)

#############################
### Inicializar WandB
#############################
# Cargar la clave API desde una variable de entorno
logging.info("Logging in WandB")
api_key = os.environ.get("WANDB_API_KEY")
# Iniciar sesión en W&B
wandb.login(key=api_key)

# create a wandb run
run = wandb.init(project="Swin_UPENN", job_type="train", config=config_train)

# we pass the config back from W&B
config_train = wandb.config


directory = "./Dataset"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


### Setup average meter, fold reader, checkpoint saver
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


###############################
#### Compose functions
###############################
train_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        ConvertToMultiChannel_with_infiltration(keys="label"),
        transforms.CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[roi[0], roi[1], roi[2]],
        ),
        transforms.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[roi[0], roi[1], roi[2]],
            random_size=False,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        ConvertToMultiChannel_with_infiltration(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# Create Swin transformer
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=roi,
    in_channels=11,
    out_channels=3,  # mdificar con edema
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

# Optimiser function loss
torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(
    include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=infer_overlap,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


# Define Train and Validation Epoch
def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            # # wandb
            # wandb.log(
            #     {
            #         "val_dice_nroi": dice_tc,
            #         "val_dice_froi": dice_wt,
            #         "val_dice_edema": dice_et,
            #     }
            # )
            start_time = time.time()

    return run_acc.avg


# Define training
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        # wandb
        print("train_loss", train_loss.item(), type(train_loss.item()))
        print(
            "lr", optimizer.param_groups[0]["lr"], type(optimizer.param_groups[0]["lr"])
        )

        wandb.log(
            {
                "loss": train_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            # wandb
            wandb.log(
                {
                    "val_dice_nroi": dice_tc,
                    "val_dice_froi": dice_wt,
                    "val_dice_edema": dice_et,
                    "val_dice_avg": val_avg_acc,
                }
            )
            start_time = time.time()
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            # dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                )

            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    # Guardar artefacto en W&B
    artifact_name = f"{wandb.run.id}_best_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(os.path.join(directory, "model.pt"))
    wandb.log_artifact(at, aliases=["final"])

    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


####################################
# Load DATASET and training modelo #
####################################
def main(config_train):
    dataset_path = "./Dataset/Dataset_30_casos/"

    train_set = CustomDataset(
        dataset_path, section="train", transform=train_transform
    )  # t_transform
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    im_t = train_set[0]
    # (im_t["image"].shape)
    print(im_t["image"].shape)

    val_set = CustomDataset(
        dataset_path, section="valid", transform=val_transform
    )  # v_transform
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    im_v = val_set[0]
    # (im_t["image"].shape)
    print(im_v["label"].shape)
    # return
    # im_v = val_set[0]
    # print(im_v["image"].shape)
    # print(im_v["label"].shape)

    ##########################################################
    # Comenzar entrenamiento
    ##########################################################
    start_epoch = 0

    (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    print(f"train completed, best average dice: {val_acc_max:.4f} ")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, loss_epochs, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_avg, color="green")
    plt.show()
    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_tc, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_wt, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    plt.xlabel("epoch")
    plt.plot(trains_epoch, dices_et, color="purple")
    plt.show()

    # finish W&B run


if directory is None:
    shutil.rmtree(root_dir)

if __name__ == "__main__":
    main(config_train)
