import torch


def mask_graph_batch(batch, mask_ratio=0.15, mask_specie_ratio=0.5):

    mask = torch.zeros_like(batch.atomic_numbers, dtype=torch.bool)

    flag = (mask_specie_ratio > torch.rand(1)) # chance to mask out one specie
    if flag:
        for i in range(len(batch.natoms)):
            local_mask = torch.zeros(batch.natoms[i], dtype=torch.bool).to(batch.natoms.device)
            local_atoms = batch.atomic_numbers[batch.batch == i]
            chosen_idx = torch.randint(0, batch.natoms[i]-1, (1,))
            mask_idx = (local_atoms==local_atoms[chosen_idx.item()]).to(batch.natoms.device)
            local_mask[mask_idx] = True
            mask[batch.batch == i] = local_mask
    else:
        mask_nums = torch.ceil(batch.natoms * mask_ratio).int()
        for i in range(len(batch.natoms)):
            local_mask = torch.zeros(batch.natoms[i], dtype=torch.bool).to(batch.natoms.device)
            mask_idx = torch.randint(0, batch.natoms[i]-1, (mask_nums[i],)).to(batch.natoms.device)
            local_mask[mask_idx] = True
            mask[batch.batch == i] = local_mask

    return mask
