#!/usr/bin/env python3


import torch
from train import CNN
from transformers import AutoConfig


# Step 2: Load the model and save the state dictionary
model = CNN("../data/")
model.load_state_dict(torch.load("../models/the_model.bin"))

torch.save(model.state_dict(), "../models/pytorch_model.bin")
