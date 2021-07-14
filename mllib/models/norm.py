import torch.nn as nn

def get_norm_layer(ch_size: int, norm_type: str = 'bn', group_size: int = 2):
    assert norm_type in ["BN", "GN", "LN"], f"Unsupported norm_type: {norm_type}"

    norm = None
    if norm_type == "BN":
        norm = nn.BatchNorm2d(ch_size)
    elif norm_type == 'GN':
        norm = nn.GroupNorm(group_size, ch_size)
    elif norm_type == 'LN':
        norm = nn.GroupNorm(1, ch_size)

    return norm