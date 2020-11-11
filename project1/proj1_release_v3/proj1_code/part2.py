import torch

def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, c)
    - filter: Torch tensor of shape (k, j)
    Returns
    - filtered_image: Torch tensor of shape (m, n, c)
    HINTS:
    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.nn.functional.pad
    """
    filtered_image = torch.Tensor()

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    # Find the image and filter size
    m = image.size()[0]
    n = image.size()[1]
    c = image.size()[2]

    k = filter.size()[0]
    j = filter.size()[1]

    # Create empty filtered image tensor
    filtered_image = torch.zeros(m,n,c)

    # Create and implement zero padding
    i = k // 2 # number of paddings to be implemented
    u = j // 2
    padded_image = torch.nn.functional.pad(input=image, pad=(0, 0, u, u, i, i), mode='constant', value=0)

    # Implement filtered signal
    for r in range(c):
        for p in range(m):
            for q in range(n):
                frac = padded_image[p:p+k, q:q+j, r] # Dummy tensor
                filtered_image[p, q, r] = torch.sum(frac*filter) # sum of values from element-wise multiplication

    ############################################################################

    #raise NotImplementedError('`my_imfilter` function in `part2.py` ' +
    #                          'needs to be implemented')

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    return filtered_image


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args
    - image1: Torch tensor of dim (m, n, c)
    - image2: Torch tensor of dim (m, n, c)
    - filter: Torch tensor of dim (x, y)
    Returns
    - low_frequencies: Torch tensor of shape (m, n, c)
    - high_frequencies: Torch tensor of shape (m, n, c)
    - hybrid_image: Torch tensor of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping' ('clamping' in torch).
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    hybrid_image = torch.Tensor()
    low_frequencies = torch.Tensor()
    high_frequencies = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    # TODO: YOUR CODE HERE
    # Find the image and filter size
    m = image1.size()[0]
    n = image1.size()[1]
    c = image1.size()[2]

    k = filter.size()[0]
    j = filter.size()[1]

    # Create empty filtered image tensor
    low_frequencies1 = torch.zeros(m,n,c)
    low_frequencies2 = torch.zeros(m,n,c)
    hybrid_image = torch.zeros(m,n,c)

    # Create and implement zero padding
    i = k // 2 # number of paddings to be implemented
    padded_image1 = torch.nn.functional.pad(input=image1, pad=(0, 0, i, i, i, i), mode='constant', value=0)
    padded_image2 = torch.nn.functional.pad(input=image2, pad=(0, 0, i, i, i, i), mode='constant', value=0)

    # Implement filtered signal
    for r in range(c):
        for p in range(m):
            for q in range(n):
                frac1 = padded_image1[p:p+k, q:q+k, r] # Dummy tensor
                frac2 = padded_image2[p:p+k, q:q+k, r] # Dummy tensor
                low_frequencies1[p, q, r] = torch.sum(frac1*filter) # sum of values from element-wise multiplication
                low_frequencies2[p, q, r] = torch.sum(frac2*filter) # sum of values from element-wise multiplication

    # Obtain low_frequencies from filtered result of image 1
    low_frequencies = low_frequencies1

    # Obtain high_frequencies from subtracting low pass filtered result of image2 from original image
    high_frequencies = image2 - low_frequencies2

    # Obtain hybrid image by
    hybrid_image_pre_clamped = low_frequencies + high_frequencies
    hybrid_image = torch.clamp(hybrid_image_pre_clamped, 0, 1)

    ############################################################################

    #raise NotImplementedError('`create_hybrid_image` function in ' +
    #                          '`part2.py` needs to be implemented')

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    return low_frequencies, high_frequencies, hybrid_image
