import cv2
import torch
import segmentation_models_pytorch as smp
import numpy as np
import tqdm
import os
import albumentations
from PIL import Image
import aitool.metrics as metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import wandb


def Train(model: torch.nn.Module, data_dir: str, val_data_dir: str = "", batch_size: int = 1,
          epochs=100,
          out_channel=21,
          train_transform=None,
          val_transform=None,
          learn_rate=0.0001,
          num_workers=0,
          labels_name=None,
          pred_dir="./result",
          project_name="segment",
          eval_interval=1,
          save_image=False
          ):
    r"""
    This function will convert the input image to RGB (channel=3)

    data_dir format:
        images
            0.png
        masks
            0.png
    """
    beta = 1
    run = wandb.init(project=project_name)
    config = wandb.config
    config.learning_rate = learn_rate

    class ds(torch.utils.data.Dataset):
        def __init__(self, data_dir: str, transform=None, out_channal=21):
            self.images = [os.path.join(data_dir, 'images', i) for i in os.listdir(
                os.path.join(data_dir, 'images'))]
            self.masks = [os.path.join(data_dir, 'masks', i) for i in os.listdir(
                os.path.join(data_dir, 'masks'))]
            self.images.sort()
            self.masks.sort()
            self.len = len(self.images)
            self.src_shape = [0] * self.len
            self.transform = transform
            self.out_channal = out_channal
            assert self.len == len(self.masks)

        def __getitem__(self, index):
            image = Image.open(self.images[index]).convert("RGB")
            mask = Image.open(self.masks[index])
            image_np = np.array(image)
            mask_np = np.array(mask)
            mask_np[mask_np >= self.out_channal] = 255
            self.src_shape[index] = mask_np.shape
            assert len(mask_np.shape) == 2
            assert self.transform is not None
            albu = self.transform(image=image_np, mask=mask_np)
            image_albu = albu['image']
            mask_albu = albu['mask']
            image_torch = torch.from_numpy(image_albu).float()
            mask_torch = torch.from_numpy(mask_albu).long()
            image_torch = image_torch.permute(2, 0, 1)
            return image_torch, mask_torch

        def mask(self, index):
            mask = Image.open(self.masks[index])
            mask_np = np.array(mask)
            return mask_np

        def image(self, index):
            image = Image.open(self.images[index]).convert("RGB")
            image_np = np.array(image)
            return image_np

        def __len__(self):
            return self.len

    os.makedirs(pred_dir, exist_ok=True)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = ds(data_dir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if len(val_data_dir) > 0:
        val_data = ds(val_data_dir, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=1, shuffle=False, num_workers=0)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=epochs, gamma=0.1)
    model.to(dev)
    tbar = tqdm.tqdm(range(epochs))
    for epoch in tbar:
        model.train()
        r_loss = 0
        for data in train_loader:
            image_torch, mask_torch = data
            output = model(image_torch.to(dev))
            loss = loss_func(output, mask_torch.to(dev))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
            pass
        lr_scheduler.step()
        r_miou = 0
        if epoch % eval_interval > 0:
            continue
        if len(val_data_dir) > 0:
            model.eval()
            hist = np.zeros((out_channel, out_channel))
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    image_torch, mask_torch = data
                    output = model(image_torch.to(dev))
                    pred = output.argmax(dim=1).detach().cpu().numpy()[0]
                    pred = pred.astype(np.uint8)
                    pred_src = cv2.resize(pred, (int(val_data.src_shape[index][1]), int(val_data.src_shape[index][0])),
                                          interpolation=cv2.INTER_LINEAR)
                    mask_np = val_data.mask(index).astype(np.uint8)
                    hist += metrics.fast_hist(mask_np.flatten(),
                                              pred_src.flatten(), out_channel)
                    if save_image:
                        img_src = val_data.image(index).astype(np.uint8)

                        output = metrics.visual_mask(
                            pred_src, labels_name)
                        pred = metrics.visual_mask(
                            mask_np, labels_name)
                        result = np.concatenate((img_src, pred, output), axis=1)
                        cv2.imwrite(f'./{pred_dir}/{index}.png', result)
                    pass
            r_miou = np.nanmean(metrics.per_class_iu(hist))
            r_recall = np.nanmean(metrics.per_class_PA_Recall(hist))
            r_precision = np.nanmean(metrics.per_class_Precision(hist))
            r_accuracy = metrics.per_Accuracy(hist)
            f_score = (1 + beta ** 2) * r_precision * r_recall / \
                      (beta ** 2 * r_precision + r_recall)
        tbar.set_description(
            f'echo: {epoch} loss: {r_loss / len(train_loader):.4f}, miou: {r_miou:.4f}, recall: {r_recall:.4f}, accuracy: {r_accuracy:.4f}, precision: {r_precision:.4f}, f1_score: {f_score:.4f}')
        wandb.log({"loss": r_loss / len(train_loader),
                   "miou": r_miou,
                   "recall": r_recall,
                   "accuracy": r_accuracy,
                   "precision": r_precision,
                   "f1_score": f_score})
        torch.save(model.state_dict(), "model.pth")
    run.finish()
    pass


def Predict(model: torch.nn.Module, val_data_dir: str = "", result_dir: str = "rs", labels_name:list=None):
    model.eval()
    images_fs = os.listdir(val_data_dir)
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        for index, image_f in enumerate(images_fs):
            image = Image.open(os.path.join(val_data_dir, image_f)).convert("RGB")
            image_rz=image.copy().resize((1024,1024))
            image_np = np.array(image_rz)
            image_torch = torch.from_numpy(image_np).float()
            image_torch = image_torch.permute(2, 0, 1)
            image_torch = image_torch.unsqueeze(0)
            output = model(image_torch.cuda())
            pred = output.argmax(dim=1).detach().cpu().numpy()[0]
            pred = pred.astype(np.uint8)
            pred_src = cv2.resize(pred, (int(image_np.shape[1]), int(image_np.shape[0])),
                                  interpolation=cv2.INTER_LINEAR)
            pred = metrics.visual_mask(pred_src, labels_name)
            result = np.concatenate((image_np, pred), axis=1)
            cv2.imwrite(f'./{result_dir}/{index}.png', result)
            pass
    pass
