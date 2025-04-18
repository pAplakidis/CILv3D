#!/usr/bin/env python3
import torch
import torch.nn as nn

from config import *
from models.uniformer_video.uniformer import uniformer_small


class CILv3D(nn.Module):
  def __init__(
      self,
      sequence_size: int,
      state_size: int,
      command_size: int,
      filters_1d: int,
      embedding_size: int = 512,  # TODO: make dynamic (backbone_out.shape[-1])
      freeze_backbone: bool = True
    ):
    super(CILv3D, self).__init__()

    self.sequence_size = sequence_size
    self.state_size = state_size
    self.command_size = command_size
    self.filters_1d = filters_1d
    self.embedding_size = embedding_size

    uniformer_state_dict = torch.load('models/state_dicts/uniformer_small_k400_16x8.pth', map_location='cpu')
    self.uniformer = uniformer_small()
    self.uniformer.load_state_dict(uniformer_state_dict)
    self.uniformer.head = nn.Identity()

    if freeze_backbone:
      for param in self.uniformer.parameters():
        param.requires_grad = False

    # TODO: for CILv3D:
    # tf.keras.Sequential(
    #         layers=[
    #             tf.keras.layers.Conv1D(filters=self._filters_1d, kernel_size=self._1d_kernel_size, activation=self._config.activation),
    #             tf.keras.layers.BatchNormalization(),
    #             tf.keras.layers.Flatten(),
    #             tf.keras.layers.Dense(units=embeddings_size)
    #         ]
    #   )
    self.state_embedding = nn.Linear(state_size, embedding_size)
    self.command_embedding = nn.Linear(command_size, embedding_size)

    encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=TRANSFORMER_HEADS)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)

    self.linear = nn.Linear(embedding_size, 2)

  def forward(self, x, states, commands):
    """
    x: (B, C, T, H, W)
    states: (B, sequence_size, state_size)
    commands: (B, sequence_size, command_size)
    """

    state_emb = self.state_embedding(states)
    command_emb = self.command_embedding(commands)
    control_embedding = state_emb + command_emb

    # TODO: add positional embeddings (?)
    vision_embeddings, y = self.uniformer(x)
    # layerout = y[-1] # B, C, T, H, W
    # layerout = layerout[0].detach().cpu().permute(1, 2, 3, 0)

    z = vision_embeddings + control_embedding
    out = self.linear(self.transformer_encoder(z)[:, -1])
    return out


if __name__ == "__main__":
  model = CILv3D(SEQUENCE_SIZE, STATE_SIZE, COMMAND_SIZE, FILTERS_1D)
  print(model)

  # get predictions, last convolution output and the weights of the prediction layer
  vid = torch.randn(1, 3, SEQUENCE_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1])  # B, C, T, H, W
  states = torch.randn(1, SEQUENCE_SIZE, STATE_SIZE) # B, T, S
  commands = torch.randn(1, SEQUENCE_SIZE, COMMAND_SIZE) # B, T, C
  out = model(vid, states, commands)
  print(out.shape)
