# ============================================================
# SEGFORMER-B2 SECOND-STAGE FINE-TUNING
# ============================================================
#
# PURPOSE:
# Fine-tune the existing Manchester-trained SegFormer-B2 model
# on the new RoadPlugin dataset.
#
# ORIGINAL MODEL:
# D:\RoadPlugin\models\segformer_b2_final_road_model.pth
#
# IMPORTANT:
# The original model is ONLY READ.
# It is NEVER overwritten.
#
# NEW DATASET:
# D:\RoadPlugin\Segformer_Dataset
#
# NEW MODEL:
# D:\RoadPlugin\models\segformer_b2_finetuned_new_dataset_best.pth
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import random
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)

import albumentations as A


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("DEVICE INFORMATION")
print("=" * 80)

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))

    total_vram = torch.cuda.get_device_properties(
        0
    ).total_memory / (1024 ** 3)

    print(
        f"GPU VRAM: {total_vram:.2f} GB"
    )

print("Device  :", device)


# ============================================================
# PATHS
# ============================================================

# ------------------------------------------------------------
# ORIGINAL MODEL
# ------------------------------------------------------------

OLD_MODEL_PATH = (
    r"D:\RoadPlugin\models"
    r"\segformer_b2_final_road_model.pth"
)


# ------------------------------------------------------------
# NEW MODEL OUTPUT DIRECTORY
# ------------------------------------------------------------

MODEL_DIR = r"D:\RoadPlugin\models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# NEW MODEL
# ------------------------------------------------------------

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "segformer_b2_finetuned_new_dataset_best.pth"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "segformer_b2_finetuned_new_dataset_final.pth"
)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_DIR = r"D:\RoadPlugin\Segformer_Dataset"


TRAIN_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "images"
)

TRAIN_MASK_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "masks"
)


VAL_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "val",
    "images"
)

VAL_MASK_DIR = os.path.join(
    DATASET_DIR,
    "val",
    "masks"
)


TEST_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "images"
)

TEST_MASK_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "masks"
)


# ============================================================
# VERIFY ORIGINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING ORIGINAL MODEL")
print("=" * 80)

if not os.path.isfile(OLD_MODEL_PATH):

    raise FileNotFoundError(
        "Original model was not found:\n"
        + OLD_MODEL_PATH
    )

old_size_mb = (
    os.path.getsize(OLD_MODEL_PATH)
    / (1024 ** 2)
)

print("Original model found:")
print(OLD_MODEL_PATH)

print(
    f"Original model size: {old_size_mb:.2f} MB"
)

print(
    "IMPORTANT: Original model will NOT be overwritten."
)


# ============================================================
# VERIFY DATASET DIRECTORIES
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING DATASET DIRECTORIES")
print("=" * 80)

required_dirs = [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR
]

for directory in required_dirs:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            "Dataset directory not found:\n"
            + directory
        )

    print("OK:", directory)


# ============================================================
# DATASET COUNT VERIFICATION
# ============================================================

def get_files(directory, extension):

    return sorted([
        f
        for f in os.listdir(directory)
        if f.lower().endswith(extension)
    ])


train_images = get_files(
    TRAIN_IMAGE_DIR,
    ".jpg"
)

train_masks = get_files(
    TRAIN_MASK_DIR,
    ".png"
)

val_images = get_files(
    VAL_IMAGE_DIR,
    ".jpg"
)

val_masks = get_files(
    VAL_MASK_DIR,
    ".png"
)

test_images = get_files(
    TEST_IMAGE_DIR,
    ".jpg"
)

test_masks = get_files(
    TEST_MASK_DIR,
    ".png"
)


print("\n" + "=" * 80)
print("DATASET VERIFICATION")
print("=" * 80)

print(
    f"Train images : {len(train_images)}"
)

print(
    f"Train masks  : {len(train_masks)}"
)

print(
    f"Val images   : {len(val_images)}"
)

print(
    f"Val masks    : {len(val_masks)}"
)

print(
    f"Test images  : {len(test_images)}"
)

print(
    f"Test masks   : {len(test_masks)}"
)


if len(train_images) != 5048:
    raise RuntimeError(
        f"Expected 5048 training images, "
        f"found {len(train_images)}"
    )

if len(train_masks) != 5048:
    raise RuntimeError(
        f"Expected 5048 training masks, "
        f"found {len(train_masks)}"
    )

if len(val_images) != 631:
    raise RuntimeError(
        f"Expected 631 validation images, "
        f"found {len(val_images)}"
    )

if len(val_masks) != 631:
    raise RuntimeError(
        f"Expected 631 validation masks, "
        f"found {len(val_masks)}"
    )

if len(test_images) != 632:
    raise RuntimeError(
        f"Expected 632 test images, "
        f"found {len(test_images)}"
    )

if len(test_masks) != 632:
    raise RuntimeError(
        f"Expected 632 test masks, "
        f"found {len(test_masks)}"
    )

print("\n✅ DATASET COUNTS ARE PERFECT")


# ============================================================
# CHECK IMAGE/MASK MATCHING
# ============================================================

def verify_pairs(
    image_files,
    mask_directory,
    split_name
):

    missing = []

    for image_name in image_files:

        mask_name = (
            os.path.splitext(image_name)[0]
            + ".png"
        )

        mask_path = os.path.join(
            mask_directory,
            mask_name
        )

        if not os.path.isfile(mask_path):

            missing.append(
                image_name
            )

    if len(missing) > 0:

        print(
            f"\n❌ {split_name}: "
            f"{len(missing)} missing masks"
        )

        print(
            "First missing:",
            missing[:10]
        )

        raise RuntimeError(
            f"{split_name} image/mask matching failed."
        )

    print(
        f"✅ {split_name}: all image/mask pairs match"
    )


verify_pairs(
    train_images,
    TRAIN_MASK_DIR,
    "TRAIN"
)

verify_pairs(
    val_images,
    VAL_MASK_DIR,
    "VAL"
)

verify_pairs(
    test_images,
    TEST_MASK_DIR,
    "TEST"
)


# ============================================================
# SEGFORMER PROCESSOR
# ============================================================

processor = SegformerImageProcessor(
    do_resize=True,
    size={
        "height": 512,
        "width": 512
    },
    do_normalize=True,
    do_rescale=True,
    reduce_labels=False
)

print("\n✅ SegFormer processor created")


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.5
    ),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=10,
        border_mode=0,
        p=0.3
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.3
    )
])


val_transform = A.Compose([])

test_transform = A.Compose([])


# ============================================================
# DATASET CLASS
# ============================================================

class RoadDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        processor,
        transform=None
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transform = transform

        self.images = sorted([
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ])

        # ----------------------------------------------------
        # Verify mask existence
        # ----------------------------------------------------

        valid_images = []

        for image_name in self.images:

            mask_name = (
                os.path.splitext(image_name)[0]
                + ".png"
            )

            mask_path = os.path.join(
                self.mask_dir,
                mask_name
            )

            if os.path.isfile(mask_path):

                valid_images.append(
                    image_name
                )

        self.images = valid_images

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        mask_name = (
            os.path.splitext(image_name)[0]
            + ".png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name
        )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = np.array(
            Image.open(
                image_path
            ).convert("RGB")
        )

        # ====================================================
        # LOAD MASK
        # ====================================================

        mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        )

        # ====================================================
        # VERIFY SIZE
        # ====================================================

        if image.shape[:2] != (512, 512):

            raise ValueError(
                f"Wrong image size: "
                f"{image_name} -> "
                f"{image.shape[:2]}"
            )

        if mask.shape[:2] != (512, 512):

            raise ValueError(
                f"Wrong mask size: "
                f"{mask_name} -> "
                f"{mask.shape[:2]}"
            )

        # ====================================================
        # MASK CONVERSION
        #
        # PNG:
        #
        # 0   = black = background
        # 255 = white = road
        #
        # Training:
        #
        # 0 = background
        # 1 = road
        # ====================================================

        mask = (
            mask > 0
        ).astype(
            np.uint8
        )

        # ====================================================
        # AUGMENTATION
        # ====================================================

        if self.transform is not None:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]

            mask = augmented["mask"]

        # ====================================================
        # SEGFORMER PROCESSING
        # ====================================================

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"]
            .squeeze(0)
        )

        labels = (
            encoded["labels"]
            .squeeze(0)
            .long()
        )

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# ============================================================
# CREATE DATASETS
# ============================================================

print("\n" + "=" * 80)
print("CREATING DATASETS")
print("=" * 80)

train_dataset = RoadDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    processor,
    train_transform
)

val_dataset = RoadDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    processor,
    val_transform
)

test_dataset = RoadDataset(
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    processor,
    test_transform
)

print(
    "Train dataset:",
    len(train_dataset)
)

print(
    "Val dataset  :",
    len(val_dataset)
)

print(
    "Test dataset :",
    len(test_dataset)
)


# ============================================================
# CHECK ONE SAMPLE
# ============================================================

sample = train_dataset[0]

print("\n" + "=" * 80)
print("SAMPLE VERIFICATION")
print("=" * 80)

print(
    "Pixel values:",
    sample["pixel_values"].shape
)

print(
    "Labels      :",
    sample["labels"].shape
)

print(
    "Label values:",
    torch.unique(
        sample["labels"]
    )
)

if tuple(
    sample["pixel_values"].shape
) != (3, 512, 512):

    raise RuntimeError(
        "Image tensor has wrong shape."
    )

if not set(
    torch.unique(
        sample["labels"]
    ).tolist()
).issubset({0, 1}):

    raise RuntimeError(
        "Mask contains values other than 0 and 1."
    )

print(
    "\n✅ SAMPLE IS CORRECT"
)


# ============================================================
# DATALOADER
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
#
# Batch size 2 is selected for memory safety.
#
# SegFormer-B2 + 512x512 + backpropagation is memory-heavy.
# ------------------------------------------------------------

BATCH_SIZE = 2

print("\n" + "=" * 80)
print("CREATING DATALOADERS")
print("=" * 80)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Val batches:",
    len(val_loader)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# LOAD EXISTING SEGFORMER-B2 ARCHITECTURE
# ============================================================

print("\n" + "=" * 80)
print("CREATING SEGFORMER-B2")
print("=" * 80)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=2,
    ignore_mismatched_sizes=True
)

print(
    "Base architecture created."
)


# ============================================================
# LOAD YOUR EXISTING MANCHESTER CHECKPOINT
# ============================================================

print("\n" + "=" * 80)
print("LOADING EXISTING MANCHESTER MODEL")
print("=" * 80)

checkpoint = torch.load(
    OLD_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    "Checkpoint type:",
    type(checkpoint)
)

if not isinstance(
    checkpoint,
    dict
):

    raise RuntimeError(
        "Unexpected checkpoint format."
    )

if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "model_state_dict was not found "
        "inside the checkpoint."
    )

old_state_dict = (
    checkpoint["model_state_dict"]
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = (
    model.load_state_dict(
        old_state_dict,
        strict=True
    )
)


print(
    "Missing keys    :",
    len(missing_keys)
)

print(
    "Unexpected keys :",
    len(unexpected_keys)
)

print(
    "Original epoch  :",
    checkpoint.get("epoch")
)

print(
    "Original best loss:",
    checkpoint.get("best_loss")
)


if len(missing_keys) != 0:

    raise RuntimeError(
        "Some model weights were not loaded."
    )

if len(unexpected_keys) != 0:

    raise RuntimeError(
        "Unexpected model weights found."
    )


print(
    "\n✅ MANCHESTER MODEL WEIGHTS LOADED PERFECTLY"
)


# ============================================================
# VERIFY MODEL CLASSES
# ============================================================

print("\n" + "=" * 80)
print("MODEL VERIFICATION")
print("=" * 80)

print(
    "Number of classes:",
    model.config.num_labels
)

if model.config.num_labels != 2:

    raise RuntimeError(
        "Model does not have exactly 2 classes."
    )


classifier_weight = (
    model.decode_head.classifier.weight
)

print(
    "Classifier shape:",
    tuple(
        classifier_weight.shape
    )
)

print(
    "\nClass 0 = Background"
)

print(
    "Class 1 = Road"
)


# ============================================================
# MOVE MODEL TO GPU/CPU
# ============================================================

model = model.to(device)

print(
    "\nModel device:",
    next(
        model.parameters()
    ).device
)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        # ----------------------------------------------------
        # ROAD CLASS
        # ----------------------------------------------------

        road_probability = (
            probabilities[:, 1]
        )

        targets = targets.float()

        road_probability = (
            road_probability.reshape(
                road_probability.size(0),
                -1
            )
        )

        targets = (
            targets.reshape(
                targets.size(0),
                -1
            )
        )

        intersection = (
            road_probability * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            road_probability.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# LOSS FUNCTIONS
# ============================================================

ce_loss = nn.CrossEntropyLoss()

dice_loss = DiceLoss()

print(
    "\n✅ Cross Entropy + Dice Loss ready"
)


# ============================================================
# OPTIMIZER
# ============================================================

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\n" + "=" * 80)
print("OPTIMIZER")
print("=" * 80)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
)

print(
    "Scheduler: ReduceLROnPlateau"
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None

print(
    "AMP enabled:",
    use_amp
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_global_metrics(
    intersection,
    union,
    pred_area,
    target_area
):

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union > 0:

        iou = (
            intersection /
            union
        )

    else:

        iou = 1.0


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        pred_area +
        target_area
    )

    if dice_denominator > 0:

        dice = (
            2.0 * intersection /
            dice_denominator
        )

    else:

        dice = 1.0

    return iou, dice


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    total_intersection = 0

    total_union = 0

    total_pred_area = 0

    total_target_area = 0


    with torch.no_grad():

        for batch in loader:

            pixel_values = (
                batch["pixel_values"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            labels = (
                batch["labels"]
                .to(
                    device,
                    non_blocking=True
                )
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(
                        pixel_values=pixel_values
                    )

                    logits = F.interpolate(
                        outputs.logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                    loss_ce = ce_loss(
                        logits,
                        labels
                    )

                    loss_dice = dice_loss(
                        logits,
                        labels
                    )

                    loss = (
                        loss_ce +
                        loss_dice
                    )

            else:

                outputs = model(
                    pixel_values=pixel_values
                )

                logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss_ce = ce_loss(
                    logits,
                    labels
                )

                loss_dice = dice_loss(
                    logits,
                    labels
                )

                loss = (
                    loss_ce +
                    loss_dice
                )


            total_loss += (
                loss.item()
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = torch.argmax(
                logits,
                dim=1
            )

            pred_road = (
                predictions == 1
            )

            target_road = (
                labels == 1
            )


            # ------------------------------------------------
            # GLOBAL METRICS
            # ------------------------------------------------

            intersection = (
                pred_road &
                target_road
            ).sum().item()

            union = (
                pred_road |
                target_road
            ).sum().item()

            pred_area_batch = (
                pred_road
            ).sum().item()

            target_area_batch = (
                target_road
            ).sum().item()


            total_intersection += (
                intersection
            )

            total_union += (
                union
            )

            total_pred_area += (
                pred_area_batch
            )

            total_target_area += (
                target_area_batch
            )


    average_loss = (
        total_loss /
        len(loader)
    )


    iou, dice = calculate_global_metrics(
        total_intersection,
        total_union,
        total_pred_area,
        total_target_area
    )


    return (
        average_loss,
        iou,
        dice
    )


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 20

best_val_iou = -1.0

best_epoch = 0


history = {
    "train_loss": [],
    "val_loss": [],
    "val_iou": [],
    "val_dice": []
}


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 80)
print("STARTING SECOND-STAGE SEGFORMER FINE-TUNING")
print("=" * 80)

print(
    "Original model:",
    OLD_MODEL_PATH
)

print(
    "Training dataset:",
    DATASET_DIR
)

print(
    "Best output:",
    BEST_MODEL_PATH
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Epochs:",
    EPOCHS
)

print("=" * 80)


for epoch in range(EPOCHS):


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()

    running_train_loss = 0.0


    progress_bar = tqdm(
        train_loader,
        desc=(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )
    )


    for batch in progress_bar:


        pixel_values = (
            batch["pixel_values"]
            .to(
                device,
                non_blocking=True
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True
            )
        )


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD + LOSS
        # ----------------------------------------------------

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    pixel_values=pixel_values
                )

                logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss_ce = ce_loss(
                    logits,
                    labels
                )

                loss_dice = dice_loss(
                    logits,
                    labels
                )

                loss = (
                    loss_ce +
                    loss_dice
                )


            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()


            # ------------------------------------------------
            # UPDATE
            # ------------------------------------------------

            scaler.step(
                optimizer
            )

            scaler.update()


        else:

            outputs = model(
                pixel_values=pixel_values
            )

            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss_ce = ce_loss(
                logits,
                labels
            )

            loss_dice = dice_loss(
                logits,
                labels
            )

            loss = (
                loss_ce +
                loss_dice
            )


            loss.backward()

            optimizer.step()


        # ----------------------------------------------------
        # TRAIN LOSS
        # ----------------------------------------------------

        running_train_loss += (
            loss.item()
        )


        progress_bar.set_postfix(
            loss=(
                f"{loss.item():.4f}"
            )
        )


    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    val_loss, val_iou, val_dice = (
        evaluate_model(
            model,
            val_loader
        )
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_iou
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "val_loss"
    ].append(
        val_loss
    )

    history[
        "val_iou"
    ].append(
        val_iou
    )

    history[
        "val_dice"
    ].append(
        val_dice
    )


    # ========================================================
    # CURRENT LEARNING RATE
    # ========================================================

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print("=" * 80)

    print(
        f"EPOCH "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.6f}"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.6f}"
    )

    print(
        f"Val IoU    : "
        f"{val_iou:.6f}"
    )

    print(
        f"Val Dice   : "
        f"{val_dice:.6f}"
    )

    print(
        f"Learning Rate : "
        f"{current_lr:.2e}"
    )

    print("=" * 80)


    # ========================================================
    # SAVE BEST MODEL
    #
    # IMPORTANT:
    # This creates a NEW file.
    #
    # It never writes to OLD_MODEL_PATH.
    # ========================================================

    if val_iou > best_val_iou:

        best_val_iou = val_iou

        best_epoch = (
            epoch + 1
        )


        torch.save(
            {
                "epoch": (
                    epoch + 1
                ),

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_iou":
                    best_val_iou,

                "val_loss":
                    val_loss,

                "val_dice":
                    val_dice,

                "learning_rate":
                    current_lr
            },
            BEST_MODEL_PATH
        )


        print("\n")
        print(
            "⭐ NEW BEST MODEL SAVED"
        )

        print(
            "Best validation IoU:",
            f"{best_val_iou:.6f}"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Saved to:",
            BEST_MODEL_PATH
        )


    # ========================================================
    # GPU MEMORY CLEANUP
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVING FINAL MODEL")
print("=" * 80)


torch.save(
    {
        "epoch": EPOCHS,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_iou":
            best_val_iou,

        "best_epoch":
            best_epoch
    },
    FINAL_MODEL_PATH
)


print(
    "Final model saved:"
)

print(
    FINAL_MODEL_PATH
)


# ============================================================
# LOAD BEST MODEL FOR TESTING
# ============================================================

print("\n" + "=" * 80)
print("LOADING BEST FINE-TUNED MODEL")
print("=" * 80)


best_checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)


model = model.to(device)


print(
    "Best epoch:",
    best_checkpoint[
        "epoch"
    ]
)

print(
    "Best validation IoU:",
    best_checkpoint[
        "best_val_iou"
    ]
)

print(
    "Best validation Dice:",
    best_checkpoint[
        "val_dice"
    ]
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

test_loss, test_iou, test_dice = (
    evaluate_model(
        model,
        test_loader
    )
)


print(
    f"Test Loss : "
    f"{test_loss:.6f}"
)

print(
    f"Test IoU  : "
    f"{test_iou:.6f}"
)

print(
    f"Test Dice : "
    f"{test_dice:.6f}"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)

print(
    "Original model:"
)

print(
    OLD_MODEL_PATH
)

print(
    "\nOriginal model was NOT overwritten."
)

print(
    "\nBest fine-tuned model:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nFinal fine-tuned model:"
)

print(
    FINAL_MODEL_PATH
)

print(
    "\nBest validation IoU:",
    f"{best_val_iou:.6f}"
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Final Test IoU:",
    f"{test_iou:.6f}"
)

print(
    "Final Test Dice:",
    f"{test_dice:.6f}"
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)