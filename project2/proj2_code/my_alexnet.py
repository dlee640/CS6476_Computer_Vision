import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

    ############################################################################
    # Student code begin
    num_classes = 15
    model = alexnet(pretrained = True)
    self.cnn_layers = model.features
    self.fc_layers = model.classifier
    
    self.cnn_layers[0].weight.requires_grad = False
    self.cnn_layers[0].bias.requires_grad = False
    self.cnn_layers[3].weight.requires_grad = False
    self.cnn_layers[3].bias.requires_grad = False
    self.cnn_layers[6].weight.requires_grad = False
    self.cnn_layers[6].bias.requires_grad = False
    self.cnn_layers[8].weight.requires_grad = False
    self.cnn_layers[8].bias.requires_grad = False
    self.cnn_layers[10].weight.requires_grad = False
    self.cnn_layers[10].bias.requires_grad = False

    self.fc_layers[1].weight.requires_grad = False
    self.fc_layers[1].bias.requires_grad = False
    self.fc_layers[4].weight.requires_grad = False
    self.fc_layers[4].bias.requires_grad = False

    self.fc_layers[-1] = nn.Linear(4096, num_classes)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
#################################################################

    #raise NotImplementedError('AlexNet not implemented')

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
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    temp = self.cnn_layers(x)
    temp = temp.reshape(temp.shape[0], 9216)
    model_output = self.fc_layers(temp)
    ############################################################################
    #raise NotImplementedError('forward function not implemented')

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
