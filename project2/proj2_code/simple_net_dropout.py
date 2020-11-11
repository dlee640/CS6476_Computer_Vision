import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

    ############################################################################
    # Student code begin
    self.cnn_layers = nn.Sequential(
    nn.Conv2d(1, 10, 5),
    nn.MaxPool2d(3, stride=3),
    nn.ReLU(),
    nn.Conv2d(10, 20, 5),
    nn.MaxPool2d(3, stride=3),
    nn.ReLU(),
    )
    self.fc_layers = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(500, 100),
    nn.Linear(100, 15),
    )
    ############################################################################

    #raise NotImplementedError('SimpleNetDropout not initialized')

    ############################################################################
    # Student code end
    ############################################################################

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    ############################################################################
    # Student code begin
    temp = self.cnn_layers(x)
    temp = temp.reshape(temp.shape[0], 500)
    model_output = self.fc_layers(temp)
    ############################################################################
    #raise NotImplementedError('forward function not implemented')

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
