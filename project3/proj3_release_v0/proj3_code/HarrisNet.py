#!/usr/bin/python3

from torch import nn
import torch
from typing import Tuple

import torch.nn.functional as F

from proj3_code.torch_layer_utils import (
    get_sobel_xy_parameters,
    get_gaussian_kernel,
    ImageGradientsLayer
)



class HarrisNet(nn.Module):
    """
    Implement Harris corner detector (See Szeliski 4.1.1) in pytorch by
    sequentially stacking several layers together.

    Your task is to implement the combination of pytorch module custom layers
    to perform Harris Corner detector.

    Recall that R = det(M) - alpha(trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
    http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.

    You may find the Pytorch function nn.Conv2d() helpful here.
    """

    def __init__(self):
        """
        We Create a nn.Sequential() network, using 5 specific layers (not in this
        order):
          - SecondMomentMatrixLayer: Compute S_xx, S_yy and S_xy, the output is
            a tensor of size (num_image, 3, width, height)
          - ImageGradientsLayer: Compute image gradients Ix Iy. Can be
            approximated by convolving with Sobel filter.
          - NMSLayer: Perform nonmaximum suppression, the output is a tensor of
            size (num_image, 1, width, height)
          - ChannelProductLayer: Compute I_xx, I_yy and I_xy, the output is a
            tensor of size (num_image, 3, width, height)
          - CornerResponseLayer: Compute R matrix, the output is a tensor of
            size (num_image, 1, width, height)

        To help get you started, we give you the ImageGradientsLayer layer to
        compute Ix and Iy. You will need to implement all the other layers.

        Args:
        -   None

        Returns:
        -   None
        """
        super(HarrisNet, self).__init__()

        image_gradients_layer = ImageGradientsLayer()


        # (1) ImageGradientsLayer: Compute image gradients Ix Iy. Can be
        #     approximated by convolving with sobel filter.
        # (2) EigenvalueApproxLayer: Compute S_xx, S_yy and S_xy, the output is
        #     a tensor of size num_image x 3 x width x height
        # (3) CornerResponseLayer: Compute R matrix, the output is a tensor of
        #     size num_image x 1 x width x height
        # (4) NMSLayer: Perform non-maximum suppression, the output is a tensor
        #     of size num_image x 1 x width x height

        layer_1 = ChannelProductLayer()
        layer_2 = SecondMomentMatrixLayer()
        layer_3 = CornerResponseLayer()
        layer_4 = NMSLayer()

        self.net = nn.Sequential(
            image_gradients_layer,
            layer_1,
            layer_2,
            layer_3,
            layer_4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of HarrisNet network. We will only test with 1
        image at a time, and the input image will have a single channel.

        Args:
        -   x: input Tensor of shape (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network,
            (num_image, 1, height, width) tensor

        """
        assert x.dim() == 4, \
            "Input should have 4 dimensions. Was {}".format(x.dim())

        return self.net(x)


class ChannelProductLayer(torch.nn.Module):
    """
    ChannelProductLayer: Compute I_xx, I_yy and I_xy,

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing I_xx, I_yy and I_xy respectively.
    """
    def __init__(self):
        # super(ChannelProductLayer, self).__init__()
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of the previous layer, which is of size
        (num_image x 2 x width x height) for Ix and Iy.

        Args:
        -   x: input tensor of size (num_image, 2, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for I_xx, I_yy and I_xy.

        HINT: you may find torch.cat(), torch.mul() useful here
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        
        # Separate input into Ix and Iy
        Ix_temp = x[:,0,:,:]
        Ix = torch.unsqueeze(Ix_temp, 1)
        Iy_temp = x[:,1,:,:]
        Iy = torch.unsqueeze(Iy_temp, 1)

        # Calculate Ix^2 and Iy^2
        Ixx = torch.mul(Ix, Ix)
        Ixy = torch.mul(Ix, Iy)
        Iyy = torch.mul(Iy, Iy)

        # Concatenate tensors
        output = torch.cat((Ixx,Iyy,Ixy), dim=1)
        #######################################################################

        #raise NotImplementedError('`ChannelProductLayer` need to be '
        #    + 'implemented')
        #return None

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output

class SecondMomentMatrixLayer(torch.nn.Module):
    """
    SecondMomentMatrixLayer: Given a 3-channel image I_xx, I_xy, I_yy, then
    compute S_xx, S_yy and S_xy.

    The output is a tensor of size (num_image, 3, height, width), each channel
    representing S_xx, S_yy and S_xy, respectively

    """
    def __init__(self, ksize: torch.Tensor = 7, sigma: torch.Tensor = 5):
        """
        You may find get_gaussian_kernel() useful. You must use a Gaussian
        kernel with filter size `ksize` and standard deviation `sigma`. After
        you pass the unit tests, feel free to experiment with other values.

        Args:
        -   ksize: single element tensor containing the filter size
        -   sigma: single element tensor containing the standard deviation

        Returns:
        -   None
        """
        #super(SecondMomentMatrixLayer, self).__init__()
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        self.kernel = get_gaussian_kernel(ksize, sigma)

        #######################################################################

        #raise NotImplementedError('`SecondMomentMatrixLayer` need to be '
        #    + 'implemented')
        n = self.ksize // 2
        self.conv2d_guass = nn.Conv2d(1,1,self.ksize, padding=(n,n))

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The input x here is the output of previous layer, which is of size
        (num_image, 3, width, height) for I_xx and I_yy and I_xy.

        Args:
        -   x: input tensor of size (num_image, 3, height, width)

        Returns:
        -   output: output of HarrisNet network, tensor of size
            (num_image, 3, height, width) for S_xx, S_yy and S_xy

        HINT:
        - You can either use your own implementation from project 1 to get the
        Gaussian kernel, OR reimplement it in get_gaussian_kernel().
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #

        # Separate input into Ixx, Iyy & Ixy
        Ixx_temp = x[:,0,:,:]
        Ixx = torch.unsqueeze(Ixx_temp, 1)
        Iyy_temp = x[:,1,:,:]
        Iyy = torch.unsqueeze(Iyy_temp, 1)
        Ixy_temp = x[:,2,:,:]
        Ixy = torch.unsqueeze(Ixy_temp, 1)

        # Define padding size
        n = self.ksize // 2

        # Calculate second moment matrix with padding
        Sxx = F.conv2d(Ixx, self.kernel, padding=n)
        Syy = F.conv2d(Iyy, self.kernel, padding=n)
        Sxy = F.conv2d(Ixy, self.kernel, padding=n)

        # Concatenate second moment matrices
        output = torch.cat((Sxx,Syy,Sxy), dim=1)
        
        #######################################################################

        #raise NotImplementedError('`SecondMomentMatrixLayer` needs to be '
        #    + 'implemented')
        #return None

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class CornerResponseLayer(torch.nn.Module):
    """
    Compute R matrix.

    The output is a tensor of size (num_image, channel, height, width),
    represent corner score R

    HINT:
    - For matrix A = [a b;
                      c d],
      det(A) = ad-bc, trace(A) = a+d
    """
    def __init__(self, alpha: int=0.05):
        """
        Don't modify this __init__ function!
        """
        #super(CornerResponseLayer, self).__init__()
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass to compute corner score R

        Args:
        -   x: input tensor of size (num_image, 3, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        You may find torch.mul() useful here.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        # Separate input into a,b,c,d
        a_temp = x[:,0,:,:]
        a = torch.unsqueeze(a_temp, 1)
        d_temp = x[:,1,:,:]
        d = torch.unsqueeze(d_temp, 1)
        b_temp = x[:,2,:,:]
        b = torch.unsqueeze(b_temp, 1)
        c = b

        # Calculate determinant and trace
        det = torch.mul(a,d) - torch.mul(b,c)
        trace = a + d
        alpha = 0.05
        output = det - alpha*trace**2

        #######################################################################

        #raise NotImplementedError('`CornerResponseLayer` needs to be '
        #    + 'implemented')
        #return None

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


class NMSLayer(torch.nn.Module):
    """
    NMSLayer: Perform non-maximum suppression,

    the output is a tensor of size (num_image, 1, height, width),

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    def __init__(self):
        #super(NMSLayer, self).__init__()
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum value in the neighborhood. Binarize the image
        according to locations that are equal to their maximum (i.e. 1 when it
        is equal to the maximum value and 0 otherwise), and return this binary
        image, multiplied with the cornerness response values. We'll be testing
        only 1 image at a time. Input and output will be single channel images.

        Args:
        -   x: input tensor of size (num_image, 1, height, width)

        Returns:
        -   output: the output is a tensor of size
            (num_image, 1, height, width), each representing harris corner
            score R

        (Potentially) useful functions: nn.MaxPool2d, torch.where(),
        torch.median()
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #

        # Threshold everything below median to zero
        median = torch.median(x)
        t = nn.Threshold(median.item(), 0)
        thresh = t(x)

        # Maxpooling
        m = nn.MaxPool2d(7, stride=1, padding=3)
        max = m(thresh)

        # Binarizing
        w = x.shape[2]
        h = x.shape[3]
        ones = torch.ones(w, h)
        zeros = torch.zeros(w, h)
        temp = torch.where(max == x, ones, zeros)

        # Multiply corner response score
        output = torch.mul(temp, x)

        #######################################################################

        #raise NotImplementedError('`NMSLayer` needs to be implemented')
        #return None

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return output


def get_interest_points(image: torch.Tensor, num_points: int = 3000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to return top most N x,y points with the highest confident corner
    score. Note that the return type should be tensors. Also make sure to
    sort them in descending order of confidence!

    (Potentially) useful functions: torch.nonzero, torch.masked_select,
    torch.argsort

    Args:
    -   image: A tensor of shape (b,c,m,n). We will provide an image of
        (c = 1) for grayscale image.

    Returns:
    -   x: A tensor array of shape (N,) containing x-coordinates of
        interest points
    -   y: A tensor array of shape (N,) containing y-coordinates of
        interest points
    -   confidences: tensor array of dim (N,) containing the
        strength of each interest point
    """

    # We initialize the Harris detector here, you'll need to implement the
    # HarrisNet() class
    harris_detector = HarrisNet()

    # The output of the detector is an R matrix of the same size as image,
    # indicating the corner score of each pixel. After non-maximum suppression,
    # most of R will be 0.
    R = harris_detector(image)

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    # Normalize c matrix
    maxval = torch.max(R)
    c = R / maxval
    c = torch.squeeze(c)

    # Flattening the nonzero x_index, y_index, & confidence values into 1D tensor
    x_temp = torch.flatten(c.nonzero()[:, 0])
    y_temp = torch.flatten(c.nonzero()[:, 1])
    c_temp = torch.flatten(c[c.nonzero()[:, 0],c.nonzero()[:, 1]])
    num = len(c_temp)

    # Remove the border values
    y,x,c= remove_border_vals(image, x_temp, y_temp, c_temp)

    # Sort x,y,c in descending order
    ind = torch.argsort(c, descending=True)
    confidences = c[ind]
    x = x[ind]
    y = y[ind]

    # Filter out number of points if too many
    if num_points < num:
        x = x[:num_points]
        y = y[:num_points]
        confidences = confidences[:num_points]

    ###########################################################################

    #raise NotImplementedError('`get_interest_points` in `HarrisNet.py needs `'
    #    + 'be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


    return x, y, confidences



def remove_border_vals(img, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a 16x16 window around
    that point does not lie completely within the input image.

    Note: maintain the ordering which is input to this function.

    Args:
    -   x: Torch tensor of shape (M,)
    -   y: Torch tensor of shape (M,)
    -   c: Torch tensor of shape (M,)

    Returns:
    -   x: Torch tensor of shape (N,), where N <= M (less than or equal after
        pruning)
    -   y: Torch tensor of shape (N,)
    -   c: Torch tensor of shape (N,)
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    x_lim = img.shape[2]
    y_lim = img.shape[3]

    # Filter out left x borders
    xo = x
    yo = y
    co = c

    x = x[xo > 16]
    y = y[xo > 16]
    c = c[xo > 16]

    # Filter out right x borders
    xo = x
    yo = y
    co = c

    x = x[xo < (x_lim - 16)]
    y = y[xo < (x_lim - 16)]
    c = c[xo < (x_lim - 16)]

    # Filter out top y borders
    xo = x
    yo = y
    co = c

    x = x[yo > 16]
    y = y[yo > 16]
    c = c[yo > 16]

    # Filter out bottom y borders
    xo = x
    yo = y
    co = c

    x = x[yo < (y_lim - 16)]
    y = y[yo < (y_lim - 16)]
    c = c[yo < (y_lim - 16)]


    ###########################################################################

    #raise NotImplementedError('`remove_border_vals` in `HarrisNet.py` needs '
    #    + 'to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return x, y, c


