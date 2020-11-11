import torch


def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference: 
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other 
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation 
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,)
    """
    filtered_signal = torch.FloatTensor()

    #############################################################################
    # TODO: YOUR CODE HERE

    # Find the signal & kernel size
    N = signal.size()[0]
    K = kernel.size()[0]

    # Define size of filtered signal
    filtered_signal = torch.zeros(N)

    # Create and implement zero padding
    number_of_padding = K // 2
    zero_padding = torch.zeros(number_of_padding)
    padded_signal = torch.cat((zero_padding, signal, zero_padding), dim=0)
    
    # Implement filtered signal
    for i in range(N):
        frac = torch.zeros(K) # Dummy tensor
        for j in range(K):
            frac[j] = padded_signal[i + j]
        filtered_signal[i] = torch.dot(frac, kernel)


    ############################################################################
    
    #raise NotImplementedError
    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return filtered_signal
