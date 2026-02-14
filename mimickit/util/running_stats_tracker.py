import torch

class RunningStatsTracker():
    def __init__(self, num_entries, window_size, device):
        self._window_size = window_size
        self._vals = torch.zeros([num_entries, window_size], device=device, dtype=torch.float32)
        self._counts = torch.zeros([num_entries], device=device, dtype=torch.int)
        self._heads = torch.zeros([num_entries], device=device, dtype=torch.int)
        self._means = torch.zeros([num_entries], device=device, dtype=torch.float32)
        self._need_update = torch.zeros([num_entries], device=device, dtype=torch.bool)
        self._has_new_vals = False
        return

    def reset(self):
        self._vals[:] = 0
        self._counts[:] = 0
        self._heads[:] = 0
        self._means[:] = 0
        self._need_update[:] = False
        self._has_new_vals = False
        return

    def calc_means(self):
        if (self._has_new_vals):
            self._update_stats()
            self._has_new_vals = False
        
        return self._means

    def update(self, indices, vals):
        n = indices.shape[0]
        if (n > 0):
            self._has_new_vals = True
            
        for i in range(n):
            curr_idx = indices[i]
            curr_val = vals[i]

            curr_head = self._heads[curr_idx]
            self._vals[curr_idx, curr_head] = curr_val
            self._heads[curr_idx] = torch.remainder(curr_head + 1, self._window_size)
            self._need_update[curr_idx] = True
        return
    
    def _update_stats(self):
        new_vals = self._vals[self._need_update]
        new_means = torch.mean(new_vals, dim=-1)
        self._means[self._need_update] = new_means
        return