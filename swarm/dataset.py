
import torch 
from torch.utils.data import Dataset 
from torchvision import transforms

import einops
from typing import Iterable
import random


def gen_contiguous_mask(length, starts, width=1, dtype=torch.float32):
    """Generates a binary mask with contiguous sets of zeros

    Example:
        length = 10         # Length of mask vector
        starts = [1, 5, 6]  # Start of zero-segments
        width = 2           # Size of zero-segments
    Will result in:
        mask = [1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

    Note that the zero-segments [5, 6] and [6, 7] overlaps

    Parameters
    ----------
    length: int
        Length of resulting mask vector
    starts: torch.Tensor[int]
        Start indices of zero-segments
    width: int
        Size of the zero-segments
    dtype: torch datatype
        Datatype of output mask

    Returns
    -------
    mask : torch.Tensor<length>
    """
    # Generate indices for the positions that will be masked
    indices = torch.reshape(starts[:, None] + torch.arange(0, width)[None],
                            (-1, 1)).to(torch.int64)
    updates = torch.ones(starts.shape[0] * width)
    hits = torch.zeros([length])
    # "Paint" in the updates in an empty (zero) array
    hits = hits.scatter_(0, indices[:,0], updates)
    # The mask should be True/1 wherever nothing was "painted"
    return (hits==0).to(dtype)


def mask_tensor(x, mask, axis=0):
    """Masks out a tensor with a provided mask along a given axis

    Example:
        x = [[0, 1, 2, 3],  # Mask this tensor
             [4, 5, 6, 7]]
        mask = [0, 1]       # Using this mask
        axis = 1            # Along this axis (columns)
    Will result in:
        y = [[0, 0, 0, 0],
             [1, 1, 1, 1]]

    Parameters
    ---------
    x : torch.Tensor<...>
        The tensor to mask (must have rank > 0)
    mask : torch.Tensor<?>
        Multiplicative mask vector (must match x along masking axis)
    axis : int
        The axis to apply the mask along

    Returns
    -------
    torch.Tensor<...>
    """
    shape = x.shape
    length = shape[axis]
    rank = len(shape)
    new_shape = [length if i==axis else 1 for i in range(rank)]
    mask = torch.reshape(mask, new_shape)
    return x * mask

def sample_n_choose_k(n, k):
    """Samples k elements from [0...n) or range without replacement

    Parameters
    ----------
    n : int or tuple(int, int)
        Either max value to sample or start and end index to sample from
    k : int

    Returns
    -------
    torch.Tensor<k>
    """
    if isinstance(n, Iterable):
        return torch.tensor(random.sample(range(*n), k))
    return torch.tensor(random.sample(range(n), k))

def randomly_mask_tensor(x, p, width=1, axis=0, prob=1.0, limit_range=None, flip_prob=0.0):
    """Zero out random slices along an axis

    See `mask_tensor` except with a randomized mask
    See also `gen_contiguous_mask`

    Parameters
    ----------
    x : torch.Tensor<...>
        Input tensor
    p : float
        Fraction of values to zero out
    width: int
        Number of contiguous zeros in the mask
    axis: int
        Tensor axis to swap along
    prob: float
        Probability of masking. 1.0 means:
        always masking applied (default=1.0)
    limit_range: tuple(int, int) or None
        Range allowed to be masked. None means:
        limit_range=x.shape[axis]-width (default is None)
    flip_prob: float
        Probability of flipping the mask with torch.flip (default is 0.0)

    Returns
    -------
    torch.Tensor<...>
    torch.Tensor<length>
    """
    v = torch.rand(())
    total_mask = torch.ones(x.shape[axis])
    if v < prob:
        shape = x.shape
        if limit_range is None:
            length = shape[axis]
            n = length - width
        else:
            n = limit_range.copy()
            n[1] = n[1] - width
            length = n[1] - n[0]
        ratio = length / width
        num_starts = int(p * ratio)
        starts = sample_n_choose_k(n=n, k=num_starts)
        mask = gen_contiguous_mask(shape[axis], starts, width)
        if torch.rand(()) < flip_prob:
            mask = torch.flip(mask, dims=(0,))
        x = mask_tensor(x, mask=mask, axis=axis)
        total_mask *= mask
    return x, total_mask

def get_batch(x, configs):
  # x = torch.from_numpy(sub_df[cols].to_numpy())
  x = einops.rearrange(x, 'S C -> C S') # [6, 396]
  x = torch.stft(
      input=x,
      n_fft=configs['n_fft'], # 33
      hop_length=configs['hop_length'], # 17
      win_length=configs['n_fft'], # 33
      window=torch.hann_window(configs['n_fft']), # 33
      center=False,
      return_complex=True
  ) 
  x = torch.stack([torch.real(x), torch.imag(x)], -1)
  x = torch.abs(x)

  # NOTE: NEW 
  # log_magnitude_spectrum = torch.log1p(x)
  # q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
  # threshlds = torch.quantile(log_magnitude_spectrum, q, dim=2)
  # b, c, f, p = threshlds.shape
  # q1 = threshlds[0].view(c, f, 1, p).repeat(1, 1, 22, 1)
  # low_amplitude_mask = log_magnitude_spectrum <= q1
  # NOTE: NEW 

  x = einops.rearrange(x, 'C F T P -> T C F P') # 45 6 17 2
  x = x.type(torch.float32)

  return x



class RQADataset(Dataset):
  def __init__(self, configs, windows, labels=None, mode="train") -> None:
    super().__init__()
    self.windows = windows
    self.cols = ['accelUserXFiltered', 'accelUserYFiltered', 'accelUserZFiltered', 'gyroXFiltered', 'gyroYFiltered', 'gyroZFiltered']
    self.configs = configs
    self.transform = torch.nn.Sequential(
                        transforms.Normalize((0.), (1.)),
                    )
    self.mode = mode
    self.labels = labels

    self.label_dict = {
       'smooth': 0,
       'short_distress': 1,
       'long_distress': 2

    }
  
  def __len__(self):
    return len(self.windows)

  def __getitem__(self, idx):
    x = torch.from_numpy(self.windows[idx][self.cols].to_numpy())
    if self.mode == "test":
        label = self.label_dict[self.labels[idx]]
    else:
       label = -1


    # compute STFT
    x = get_batch(x, self.configs)

    # Normalize
    x_norm = (x - x.mean(0, keepdim=True))/(x.std(0, keepdim=True) + 1e-8)

    # Target
    y = x_norm.clone()

    # Mask out
    loss_mask = torch.ones_like(x)
    X_time_masked, time_mask = randomly_mask_tensor(x_norm, 0.15, 3, axis=0)
    loss_mask *= einops.rearrange(time_mask, 'T -> T 1 1 1')
    X_freq_masked, freq_mask = randomly_mask_tensor(X_time_masked, 0.2, 3, axis=2)
    loss_mask *= einops.rearrange(freq_mask, 'F -> 1 1 F 1')
    loss_mask = 1.0 - loss_mask 

    # Reshape    
    x = einops.rearrange(X_freq_masked, 'T C F P -> T (C F P)')
    y = einops.rearrange(y, 'T C F P -> T (C F P)')
    loss_mask = einops.rearrange(loss_mask, 'T C F P -> T (C F P)')

    return x, (y, loss_mask, label)

    


    


