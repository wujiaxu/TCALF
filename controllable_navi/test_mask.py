import torch

# Assume `batch_S` is your input batch of state matrices with shape (batch_size, N, C)
# where batch_size is the number of samples in the batch, N is the maximum number of agents, and C is the number of features per agent.

def create_binary_mask(batch_S):
    # Step 1: Check for non-zero rows
    non_zero_rows = torch.any(batch_S != 0, dim=-1)
    
    # Step 2: Create a binary mask
    binary_mask = non_zero_rows.float()  # Optionally convert to float if needed

    return binary_mask

# Example usage
batch_size = 4
N = 5  # maximum number of agents
C = 3  # number of features per agent

# Example batch of state matrices (batch_size, N, C)
batch_S = torch.tensor([
    [[1, 2, 3,4, 5, 6,0, 0, 0,0, 0, 0,0, 0, 0]],
    [[1, 0, 1,0, 2, 3,4, 5, 6,0, 0, 0,0, 0, 0]],
    [[0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0]],
    [[1, 2, 3,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0]],
])
print(batch_S.max(0))
batch_S = batch_S.reshape(4*5,-1) 
binary_mask = create_binary_mask(batch_S)
print(binary_mask)
