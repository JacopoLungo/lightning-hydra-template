import torch

def unnormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize an image tensor that was normalized with ImageNet mean and std.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        torch.Tensor: Unnormalized image tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean