import torch

def scale_pos(pos_cartesian, primitive_vecs):
    '''
    Calculate scaled positions from Cartesian positions.
    '''
    return torch.matmul(pos_cartesian, torch.inverse(primitive_vecs)).to(pos_cartesian.dtype)

def wrap_pos(pos_cartesian, primitive_vecs):
    '''
    R = P * V  
        R is positions in cartesian coordinate; 
        P is projections in primitive vectors; 
        V is primitive vectors;
    Positions wrapped into unit cell = modulo(P) * V
    '''
    projections = scale_pos(pos_cartesian, primitive_vecs).to(pos_cartesian.dtype)
    return torch.matmul(torch.remainder(projections, 1), primitive_vecs).to(projections.dtype)


def scale_batch_pos(batch_info, pos_cartesian):
    pos_scaled = (pos_cartesian.clone())
    for i in range(len(batch_info.natoms)):
        local_mask = (batch_info.batch == i)
        pos_scaled[local_mask] = scale_pos(pos_scaled[local_mask], batch_info.cell[i]).to(pos_scaled.dtype)
    return pos_scaled


def unscale_batch_pos(batch_info, pos_scaled):
    pos_cartesian = (pos_scaled.clone())
    for i in range(len(batch_info.natoms)):
        local_mask = (batch_info.batch == i)
        pos_cartesian[local_mask] = torch.matmul(pos_scaled[local_mask], batch_info.cell[i]).to(pos_scaled.dtype)
    return pos_cartesian


def add_noise_to_batch_pos(batch, sigma_ratio=0.1, distribution = 'normal', scaling = False):
    true_pos_noise = torch.zeros_like(batch.pos)
    for i in range(len(batch.natoms)):
        N = batch.natoms[i]
        local_mask = (batch.batch == i)
        primitive_vecs = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=batch.cell.dtype).to(batch.cell.device) if scaling else batch.cell[i]
        # calculate relative position difference
        relative_pos = (torch.transpose(torch.broadcast_to(batch.pos[local_mask], (N,N,3)),0,1) - batch.pos[local_mask]).to(batch.pos.device) 
        # calculate the minimum norm of relative_pos
        min_radius = torch.min(torch.norm(relative_pos, dim=-1) + torch.diag(torch.tensor([float('inf')]).repeat(N)).to(batch.pos.device)) # eliminate diagonals
        # random noise
        if distribution == 'normal':
            true_pos_noise[local_mask] = torch.normal(0, sigma_ratio * min_radius, size=(N,3)).to(batch.pos.dtype).to(batch.pos.device)
        elif distribution == 'uniform':
            true_pos_noise[local_mask] = (sigma_ratio * min_radius * 2 * (torch.rand(N,3).to(batch.pos.device) - 0.5)).to(batch.pos.dtype)
        batch.pos[local_mask] = wrap_pos(batch.pos[local_mask] + true_pos_noise[local_mask], primitive_vecs).to(batch.pos.dtype).to(batch.pos.device)
    return batch, true_pos_noise

def wrap_batch_pos(batch_info, pos_unwrapped, scaling = False):
    pos_wrapped = torch.zeros_like(batch_info.pos)
    for i in range(len(batch_info.natoms)):
        local_mask = (batch_info.batch == i)
        primitive_vecs = torch.tensor([[1,0,0],[0,1,0],[0,0,1]],dtype=batch_info.cell.dtype).to(batch_info.cell.device) if scaling else batch_info.cell[i]
        pos_wrapped[local_mask] = wrap_pos(pos_unwrapped[local_mask], primitive_vecs).to(batch_info.pos.dtype).to(batch_info.pos.device)
    return pos_wrapped