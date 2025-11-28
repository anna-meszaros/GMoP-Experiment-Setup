import numpy as np
import pandas as pd
import torch

from scipy import interpolate
from utils.memory_utils import get_total_memory, get_used_memory

# String to number mapping for agent types
# np.fromstring(np.array(['V']), dtype=np.uint32) = 86
# np.fromstring(np.array(['P']), dtype=np.uint32) = 80
# np.fromstring(np.array(['B']), dtype=np.uint32) = 66
# np.fromstring(np.array(['M']), dtype=np.uint32) = 77
# np.fromstring(np.array(['0']), dtype=np.uint32) = 48

agent_width_dict = torch.zeros(88)
agent_width_dict[[48, 66, 77, 80, 86]] = torch.tensor([0.0, 0.6, 0.8, 0.6, 1.7])

average_agent_speed_dict = torch.zeros(88)
average_agent_speed_dict[[48, 66, 77, 80, 86]] = torch.tensor([0.0, 4.0, 6.6, 1.3, 6.6])
    # 86: 6.6, #13.9,
    # 80: 1.3,
    # 66: 4.0, 
    # 77: 6.6, #13.9
    # 48: 0.0
# speed in m/s, taken from statistics in NuScenes dataset which was recorded in an urban environmen


def calculate_distance_matrix(Y):
    # Y.shape = [batch_size, n_agents, n_timesteps, n_features]

    # Only use position data
    Y_pos = Y[..., :2].float()

    # Expand the dimensions of the trajectories
    trajs_agent1 = Y_pos[:, None, :, None]    # [batch_size, 1, n_agents, 1, n_timesteps, n_features]
    trajs_agent2 = Y_pos[:, :, None, :, None] # [batch_size, n_agents, 1, n_timesteps, 1, n_features]

    # get available memory
    if isinstance(Y, torch.Tensor) and Y.device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(Y.device).total_memory
        used_memory = torch.cuda.memory_reserved(Y.device)
    else:
        total_memory, used_memory = get_total_memory(), get_used_memory()
    available_memory = total_memory - used_memory

    # Calculate required memory for distance matrix
    n_batches, n_agents, n_timesteps, _ = Y_pos.shape
    memeory_per_distance = 4 * 2 * 4 # 4 bytes per float32, 2 floats per distance, 4 parallel uses
    memory_per_sample = (n_agents * n_timesteps) ** 2 * memeory_per_distance

    # Memory for saving
    memory_for_saving = 4 * n_batches * n_agents * n_agents * n_timesteps * n_timesteps

    assert memory_for_saving < 0.8 * available_memory, "Not enough memory available for final distance matrix, requires {:0.2f} GB, only {:0.2f} GB available".format(memory_for_saving / 2 ** 30, available_memory / 2 ** 30)

    available_calculation_memory = available_memory - memory_for_saving

    if available_calculation_memory > memory_per_sample * n_batches:
        dist = torch.linalg.norm(trajs_agent1 - trajs_agent2, axis=-1)
    else:
        simultaneous_samples = int(available_calculation_memory / memory_per_sample)
        dist = torch.zeros((n_batches, n_agents, n_agents, n_timesteps, n_timesteps)).to(Y.device)
        for i in range(0, n_batches, simultaneous_samples):
            i_max = min(i + simultaneous_samples, n_batches)
            dist[i:i_max] = torch.linalg.norm(trajs_agent1[i:i_max] - trajs_agent2[i:i_max], axis=-1)

    return dist


 

def get_crossing_trajectories(Y, T, device='cpu'):
    ''' 
    Function to get the crossing trajectories between sets of trajectories

    Parameters
    ----------
    Y : torch.tensor
        Multidimensional array containing the trajectories to be compared.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    T : torch.tensor
        Array containing the types of the agents.
        Shape: [batch_size, n_agents]

    Returns
    -------
    crossing_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    crossing_class: list of torch.tensors
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs cross each other.
    '''

    n_batches = Y.shape[0]
    # Get maximum number of agents
    n_agents = Y.shape[1]

    # Get number of timesteps
    n_timesteps = Y.shape[2]

    dist = calculate_distance_matrix(Y)
    
    T_agents = torch.tile(T[:,:,None,None,None], (1, 1, n_agents, n_timesteps, n_timesteps)) # [batch_size, n_agents, n_agents, n_timesteps, n_timesteps]

    # Vectorize dictionary
    # agent_width_dict_vect = np.vectorize(agent_width_dict.get)

    # Get thresholds depending on the average width of the agents
    # thresholds = agent_width_dict_vect(T_agents)
    thresholds = agent_width_dict.to(device)[T_agents.int()]
    del T_agents

    # Get where agents are close to each other depending on average class width
    d1 = dist < thresholds
    del thresholds
    # only get the upper diagonal of the matrix w.r.t. the agents
    # agent pairs are stacked within each batch
    triu_indices = torch.triu_indices(n_agents, n_agents, offset=1)
    d1_pairs_perBatch = d1[:,triu_indices[0],triu_indices[1],:,:]

    d1_pairs = d1_pairs_perBatch.reshape(-1, n_timesteps, n_timesteps)

    agent_ids_gen = torch.cat([triu_indices[0][:,None], triu_indices[1][:,None]], dim=-1)

    # Get the crossing trajectories
    crossing_class_gen = torch.zeros((n_batches, n_agents*(n_agents-1)))

    num_agent_pairs = triu_indices[0].shape[0]

    for i in range(len(d1_pairs)):
        if torch.any(d1_pairs[i]):
            first_intersecting_timesteps = torch.unravel_index(torch.argmax(d1_pairs[i].int()), d1_pairs[i].shape)
            if first_intersecting_timesteps[0] < first_intersecting_timesteps[1]:
                crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 1

            elif first_intersecting_timesteps[0] > first_intersecting_timesteps[1]:
                crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 2
        else:
            crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 0


    # create duplicate array of the crossing_class array
    crossing_class_copy = crossing_class_gen[:,:num_agent_pairs].clone()
    # change the crossing classes around so that there are examples with the swapped agent order
    crossing_class_copy[crossing_class_gen[:,:num_agent_pairs] == 1] = 2
    crossing_class_copy[crossing_class_gen[:,:num_agent_pairs] == 2] = 1

    crossing_class_gen[:, num_agent_pairs:] = crossing_class_copy

    agent_ids_gen = torch.cat([agent_ids_gen, torch.cat([triu_indices[1][:,None], triu_indices[0][:,None]], axis=-1)])

    # remove nan entries
    existing_agents = ~torch.isnan(Y).any(axis=(2,3))

    crossing_agent_ids = []
    crossing_class = []

    for i in range(n_batches):
        mask_0 = existing_agents[i, agent_ids_gen[:,0]]
        mask_1 = existing_agents[i, agent_ids_gen[:,1]]

        mask = mask_0 & mask_1

        crossing_agent_ids.append(agent_ids_gen[mask.cpu()].numpy())
        crossing_class.append(crossing_class_gen[i, mask.cpu()].numpy())
    

    return crossing_agent_ids, crossing_class

def propagate_last_seen_value(arr1, arr2, init_val):

    last_value = init_val
    for i in range(len(arr2)):
        if arr2[i] != 0:
            last_value = arr1[i]  # Update last seen non-zero value
        else:
            arr1[i] = last_value  # Overwrite with the last seen value

    return arr1


def get_hypothetical_path_crossing(X, Y, T, dt, device='cpu'):

    ''' 
    Function to get the hypothetical crossing trajectories between sets of trajectories

    Parameters
    ----------
    X_t0: torch.tensor
        Multidimensional array containing trajectory info at the last observed timestep.
        Shape: [batch_size, n_agents, n_features]

    Y : torch.tensor
        Multidimensional array containing the future trajectories to be used as based for comparison.
        The velocity during the future trajectory is to be modified based on the observed velocity
        at the last observation in the past.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    T : torch.tensor
        Array containing the types of the agents.
        Shape: [batch_size, n_agents]

    dt: float
        Time step of the recorded data.

    Returns
    -------
    crossing_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    crossing_class: list of torch.tensors
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs cross each other.
    '''

    n_batches, n_agents, n_timesteps, _ = Y.shape

    # Vectorize dictionary
    # average_agent_speed_dict_vect = np.vectorize(average_agent_speed_dict.get)

    # Get thresholds depending on the average width of the agents
    # speed_thresholds = average_agent_speed_dict_vect(T)
    speed_thresholds = average_agent_speed_dict.to(device)[T.int()]

    traj = torch.cat([X[...,:2], Y[...,:2]], dim=2)

    X_t0 = X[..., -1, :2] - X[..., -2, :2]
    X_t0 = torch.cat([X_t0, X[..., -1, -1:]], axis=-1)

    # Get the velocity of the agents at the last timestep
    X_t0_vel = X_t0[:,:,:2]/dt

    # Get the heading of the agents at the last timestep
    X_t0_heading = X_t0[:,:,2]

    # Check if agents are moving faster than the average speed
    speed_mask = torch.linalg.norm(X_t0_vel, axis=-1) <= speed_thresholds

    # Calculate the cumulative displacements in the future
    # Y_displacements = np.insert(torch.diff(Y, axis=2),0,0, axis=2)

    Y_displacements = torch.diff(Y, dim=2)
    zeros = torch.zeros((Y_displacements.shape[0], Y_displacements.shape[1], 1, Y_displacements.shape[-1])).to(device)
    Y_displacements = torch.cat([zeros, Y_displacements], dim=2)
    Y_distances = torch.linalg.norm(Y_displacements, axis=-1)
    # set small displacements to zero as this can otherwise lead to noise in the extrapolation
    Y_distances[Y_distances < 0.05] = 0 
    Y_displacements[Y_distances == 0] = 0

    # Calculate heading at each timestep
    Y_rel = traj[...,-n_timesteps:,:2] - traj[...,-n_timesteps-1:-1,:]
    Y_heading = torch.atan2(Y_rel[:,:,:,1], Y_rel[:,:,:,0])
    # At every timestep where displacement is zero, set the heading to the heading at the previous timestep
    Y_heading = torch.stack([propagate_last_seen_value(Y_heading[b,a], Y_distances[b,a], init_val=X_t0_heading[b,a]) 
                          for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)
    # At every timestep where displacement is zero, set the displacement to 1e-3 in direction of heading
    Y_displacements[Y_distances == 0] = torch.stack([torch.cos(Y_heading), torch.sin(Y_heading)], dim=-1)[Y_distances == 0] * 1e-3
    Y_distances[Y_distances == 0] = 1e-3 

    Y_smoothed = Y[:,:,[0],:] + torch.cumsum(Y_displacements, axis=2)
    Y_smoothed = Y_smoothed.cpu().numpy()

    # Y_distances[Y_distances == 0] = 1e-3 
    Y_cumulative_distances = torch.cumsum(Y_distances, axis=-1).cpu().numpy()

    initial_speeds = torch.zeros(T.shape).to(device)
    initial_speeds[speed_mask] = speed_thresholds.to(device)[speed_mask]
    initial_speeds[~speed_mask] = torch.linalg.norm(X_t0_vel[~speed_mask], axis=-1)

    del speed_thresholds

    # Determine the new distances based on constant velocity
    new_distances = Y_distances[:,:,1:].clone()
    speed_mask = new_distances < torch.tile(initial_speeds[:,:,None], (1,1,n_timesteps-1)) * dt
    new_distances[speed_mask] = torch.tile(initial_speeds[:,:,None], (1,1,n_timesteps-1))[speed_mask] * dt

    # new_cumulative_distances = np.insert(torch.cumsum(new_distances, axis=-1), 0, 0, axis=-1)
    new_cumulative_distances = torch.cumsum(new_distances, axis=-1)
    zeros = torch.zeros((new_cumulative_distances.shape[0], new_cumulative_distances.shape[1], 1)).to(device)
    new_cumulative_distances = torch.cat([zeros, new_cumulative_distances], axis=-1).cpu().numpy()

    # Interpolate positions to match the new distances
    Y_hypothetical = torch.zeros_like(Y)

    f1 = np.stack([
            interpolate.interp1d(Y_cumulative_distances[b, a], Y_smoothed[b, a, :, 0], fill_value='extrapolate', assume_sorted=False) 
            for b in range(n_batches) 
            for a in range(n_agents)
        ]).reshape(n_batches, n_agents)
    
    f2 = np.stack([
            interpolate.interp1d(Y_cumulative_distances[b, a], Y_smoothed[b, a, :, 1], fill_value='extrapolate', assume_sorted=False) 
            for b in range(n_batches) 
            for a in range(n_agents)
        ]).reshape(n_batches, n_agents)
    
    Y_hypothetical[:,:,:,0] = torch.stack([torch.tensor(f1[b,a](new_cumulative_distances[b,a])).to(device) for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)

    Y_hypothetical[:,:,:,1] = torch.stack([torch.tensor(f2[b,a](new_cumulative_distances[b,a])).to(device) for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)
    
    crossing_agent_ids, crossing_class = get_crossing_trajectories(Y_hypothetical, T, device=device)

    return crossing_agent_ids, crossing_class


def get_closeness(Y, device='cpu'):
    '''
    Function to get the development of the closeness of agents in the future.

    Parameters
    ----------
    Y : torch.tensor
        Multidimensional array containing the trajectories to be compared.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    Returns
    -------
    closeness_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    closeness_class: list of torch.tensors
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs get closer to each other.
    '''

    n_batches = Y.shape[0]

    # Get maximum number of agents
    n_agents = Y.shape[1]

    # Get number of timesteps
    n_timesteps = Y.shape[2]

    # Expand the dimensions of the trajectories
    dist = calculate_distance_matrix(Y)

    # Get cases where agent_1 gets closer or even reaches the position of agent_2 at timestep t0 at any point of the trajectory
    d1 = dist[:,:,:,:,0] < torch.tile(dist[:,:,:,[0],0], (1,1,1,n_timesteps))
    d1 = d1.any(axis=-1)

    # Get cases where agent_2 gets closer or even reaches the position of agent_1 at timestep t0 at any point of the trajectory
    d2 = dist[:,:,:,0,:] < torch.tile(dist[:,:,:,[0],0], (1,1,1,n_timesteps))
    d2 = d2.any(axis=-1)

    # Get cases where agent_1 and agent_2 are closer at the final timestep than at the initial timestep
    d3 = dist[:,:,:,-1,-1] < dist[:,:,:,0,0]

    # Remove the diagonal of the matrix and stack agent pairs
    triu_indices = torch.triu_indices(n_agents, n_agents, offset=1)
    d1_pairs_perBatch_triu = d1[:,triu_indices[0],triu_indices[1]]
    d2_pairs_perBatch_triu = d2[:,triu_indices[0],triu_indices[1]]
    d3_pairs_perBatch_triu = d3[:,triu_indices[0],triu_indices[1]]

    tril_indices = torch.tril_indices(n_agents, n_agents, offset=-1)
    d1_pairs_perBatch_tril = d1[:,tril_indices[0],tril_indices[1]]
    d2_pairs_perBatch_tril = d2[:,tril_indices[0],tril_indices[1]]
    d3_pairs_perBatch_tril = d3[:,tril_indices[0],tril_indices[1]]

    d1_pairs = torch.cat((d1_pairs_perBatch_triu, d1_pairs_perBatch_tril), axis=-1)*1
    d2_pairs = torch.cat((d2_pairs_perBatch_triu, d2_pairs_perBatch_tril), axis=-1)*1
    d3_pairs = torch.cat((d3_pairs_perBatch_triu, d3_pairs_perBatch_tril), axis=-1)*1


    agent_ids_gen = torch.cat([triu_indices[0][:,None], triu_indices[1][:,None]], axis=-1)
    agent_ids_gen = torch.cat([agent_ids_gen, torch.cat([tril_indices[0][:,None], tril_indices[1][:,None]], axis=-1)])

    closeness_class_gen = torch.cat((d1_pairs[:,:,None], d2_pairs[:,:,None], d3_pairs[:,:,None]), axis=-1)

    # remove nan entries
    existing_agents = ~torch.isnan(Y).any(axis=(2,3))

    closeness_agent_ids = []
    closeness_class = []

    for i in range(n_batches):
        mask_0 = existing_agents[i, agent_ids_gen[:,0]]
        mask_1 = existing_agents[i, agent_ids_gen[:,1]]

        mask = mask_0 & mask_1

        closeness_agent_ids.append(agent_ids_gen[mask.cpu()].numpy())
        closeness_class.append(closeness_class_gen[i, mask].cpu().numpy())


    return closeness_agent_ids, closeness_class

    







    




