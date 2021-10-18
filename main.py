import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import configparser
import wandb
from tqdm import tqdm

class Ensemble(nn.Module):
  def __init__(self, modelA, modelB, modelA_output, modelB_output, num_classes = 5):
    super(Ensemble, self).__init__()
    self.modelA = modelA
    self.modelB = modelB

    self.modelA.fc = nn.Identity()
    self.modelB.classifier = nn.Identity()

    self.classifier = nn.Linear(modelA_output + modelB_output, num_classes)

  def forward(self, x):
    a = self.modelA(x.clone())
    a = a.view(a.size(0), -1)
    b = self.modelB(x)
    b = b.view(b.size(0), -1)
    x = torch.cat((a, b), dim=1)

    x = self.classifier(nn.ReLU()(x))
    return x

class Ensemble4(nn.Module):
  def __init__(self, modelA, modelB, modelC, modelD, num_classes = 5):
    super(Ensemble4, self).__init__()
    self.modelA = modelA
    self.modelB = modelB
    self.modelC = modelC
    self.modelD = modelD

    self.modelA.classifier = nn.Identity()
    self.modelB.classifier = nn.Identity()
    self.modelC.classifier = nn.Identity()
    self.modelD.classifier = nn.Identity()

    self.classifier = nn.Linear(1024 + 1024 + 1024 + 1024, num_classes)

  def forward(self, x):
    a = self.modelA(x.clone())
    a = a.view(a.size(0), -1)
    b = self.modelB(x.clone())
    b = b.view(b.size(0), -1)
    c = self.modelC(x.clone())
    c = c.view(c.size(0), -1)
    d = self.modelD(x.clone())
    d = d.view(d.size(0), -1)
    x = torch.cat((a, b, c, d), dim=1)

    x = self.classifier(nn.ReLU()(x))
    return x

def main():
  config = configparser.ConfigParser()
  config.read("config.cfg")
  manual_seed(int(config["common"]["random_seed"]))

  wandb.init(project="flowers5")
  wandb.config.update(dict(config["train"]))
  wandb.config.update({ "batch_size": config["dataset"]["batch_size"] })

  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  dataset = torchvision.datasets.ImageFolder(root=config["dataset"]["path"], transform=transform)
  dataset_length = len(dataset)
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [math.floor(dataset_length * 0.8), math.ceil(dataset_length * 0.2)])

  data_loader = {
    "train": torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=int(config["dataset"]["batch_size"]), num_workers=int(config["dataset"]["num_workers"])),
    "test": torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=int(config["dataset"]["batch_size"]), num_workers=int(config["dataset"]["num_workers"]))
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"

  if config["train"]["model"] == "resnet34":
    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(512, 5)
  elif config["train"]["model"] == "densenet121":
    net = models.densenet121(pretrained=True)
    net.classifier = nn.Linear(1024, 5)
  elif config["train"]["model"] == "ensemble_freeze":
    modelA = models.resnet34()
    modelA.fc = nn.Linear(512, 5)
    modelA.load_state_dict(torch.load("models/resnet34_best.pt")["model_state_dict"])
    modelB = models.densenet121()
    modelB.classifier = nn.Linear(1024, 5)
    modelB.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    for params in modelA.parameters():
      params.requires_grad = False
    for params in modelB.parameters():
      params.requires_grad = False

    net = Ensemble(modelA, modelB, 512, 1024, 5)
  elif config["train"]["model"] == "ensemble4_freeze":
    modelA = models.densenet121()
    modelA.classifier = nn.Linear(1024, 5)
    modelA.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    modelB = models.densenet121()
    modelB.classifier = nn.Linear(1024, 5)
    modelB.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    modelC = models.densenet121()
    modelC.classifier = nn.Linear(1024, 5)
    modelC.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    modelD = models.densenet121()
    modelD.classifier = nn.Linear(1024, 5)
    modelD.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    for params in modelA.parameters():
      params.requires_grad = False
    for params in modelB.parameters():
      params.requires_grad = False
    for params in modelC.parameters():
      params.requires_grad = False
    for params in modelD.parameters():
      params.requires_grad = False

    net = Ensemble4(modelA, modelB, modelC, modelD, 5)
  elif config["train"]["model"] == "ensemble_finetune":
    modelA = models.resnet34()
    modelA.fc = nn.Linear(512, 5)
    modelA.load_state_dict(torch.load("models/resnet34_best.pt")["model_state_dict"])
    modelB = models.densenet121()
    modelB.classifier = nn.Linear(1024, 5)
    modelB.load_state_dict(torch.load("models/densenet121_best.pt")["model_state_dict"])
    net = Ensemble(modelA, modelB, 512, 1024, 5)
  else:
    raise ValueError("unexcepted model")

  net = net.to(device)
  wandb.watch(net, log_freq=100)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(net.parameters(), lr=float(config["train"]["lr"]))
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

  num_epochs = int(config["train"]["epochs"])
  best_run = { "epoch": 0, "acc": 0.0 }
  for epoch in tqdm(range(1, num_epochs + 1)):
    print(f"Epoch {epoch}/{num_epochs}")

    for phase in tqdm(["train", "test"]):
      if phase == "train":
        net.train()
      elif phase == "test":
        net.eval()

      running_loss = 0.0
      running_corrects = 0

      for (images, labels) in tqdm(data_loader[phase]):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs = net(images)
          losses = criterion(outputs, labels)
 
          _, preds = torch.max(outputs, 1)

          if phase == "train":
            losses.backward()
            optimizer.step()
        
        loss = losses.item()
        wandb.log({ "running_loss": loss })
        running_loss += loss * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
      
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      epoch_acc = running_corrects.double() / len(data_loader[phase].dataset)
      wandb.log({ f"{phase}_acc": epoch_acc, f"{phase}_loss": epoch_loss })
      print(f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

      if phase == "test" and epoch_acc > best_run["acc"]:
        best_run = {
          "epoch": epoch,
          "acc": epoch_acc
        }

        torch.save({
          "epoch": epoch,
          "model_state_dict": net.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
        }, f"""checkpoints/{config["train"]["model"]}_{epoch}.pt""")

    scheduler.step()
  
  print(f"""Best acc: {best_run["acc"]} at epoch {best_run["epoch"]}""")
  net.load_state_dict(torch.load(f"""checkpoints/{config["train"]["model"]}_{best_run["epoch"]}.pt""")["model_state_dict"])
  torch.save({ "model_state_dict": net.state_dict() }, f"""models/{config["train"]["model"]}_best.pt""")

def manual_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

if __name__ == "__main__":
  main()