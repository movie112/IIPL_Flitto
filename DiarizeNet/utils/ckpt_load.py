import os
import torch

def ckpt_load(test_folder: str):
    assert test_folder is not None, "Please provide test_folder to load the model"
    
    ckpt_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, file))
                
    if not ckpt_files:
        raise ValueError("No checkpoint files found in the provided folder.")
    
    averaged_state = {}
    num_ckpts = len(ckpt_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_ckpt = torch.load(ckpt_files[0], map_location=device)
    if "state_dict" in first_ckpt:
        first_state = first_ckpt["state_dict"]
    else:
        first_state = first_ckpt

    for key, value in first_state.items():
        if isinstance(value, torch.Tensor):
            averaged_state[key] = value.clone() / num_ckpts
    
    for ckpt_file in ckpt_files[1:]:
        ckpt = torch.load(ckpt_file, map_location=device)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                averaged_state[key] += value / num_ckpts
    return averaged_state
