import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import numpy as np

def vis_body_pos_anim(body_pos, parents, speedup=1, fps=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    speedup = int(speedup)
    body_pos = body_pos[::speedup]
    ax, ani_obj = plot_ani(fig, ax, body_pos, parents)
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, 2.0)

    plt.show()
    return

def output_body_pos_anim(body_pos, parents, save_path, speedup=1, fps=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    speedup = int(speedup)
    body_pos = body_pos[::speedup]
    ax, ani_obj = plot_ani(fig, ax, body_pos, parents)
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    def update_view(frame_idx):
        root = body_pos[frame_idx, 0]
        ax.set_xlim(root[0] - 1, root[0] + 1)
        ax.set_ylim(root[1] - 1, root[1] + 1)
        ax.set_zlim(0, 2)

    original_update = ani_obj._func

    def update_frame(frame_idx, *args):
        original_update(frame_idx, *args)
        update_view(frame_idx)

    ani_obj._func = update_frame
    update_view(0)

    writergif = animation.PillowWriter(fps=fps)
    ani_obj.save(save_path + '.gif', writer=writergif)
    plt.close(fig)
    return
   
def plot_ani(fig, ax, body_pos, parents, fps=60, colors=None):
    links = []
    for i in range(len(parents)):
        if parents[i] != -1:
            links.append([parents[i],i])
    
    if isinstance(body_pos, torch.Tensor):
        body_pos = body_pos.cpu().detach().numpy()

    link_data = np.zeros((len(links), body_pos.shape[0]-1, 3, 2))
    xini = body_pos[0]
    if colors is None:
        color = 'r'
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color=color)[0]
                    for st,ed in links]
    elif isinstance(colors, str):
        color = colors 
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color=color)[0]
                    for st,ed in links]
    else:
        link_obj = [ax.plot([xini[st,0],xini[ed,0]],[xini[st,2],xini[ed,2]],[xini[st,1],xini[ed,1]],color=colors[j])[0]
                    for j,(st,ed) in enumerate(links)]

   
    for i in range(1, body_pos.shape[0]):
        for j,(st,ed) in enumerate(links):
            pt_st = body_pos[i-1,st] #- y_rebase
            pt_ed = body_pos[i-1,ed] #- y_rebase
            link_data[j,i-1,:,0] = pt_st
            link_data[j,i-1,:,1] = pt_ed

    def update_links(num, data_lst, obj_lst):
        cur_data_lst = data_lst[:,num,:,:] 
        cur_root = cur_data_lst[0,:,0]

        root_x = cur_root[0]
        root_y = cur_root[2]
        for obj, data in zip(obj_lst, cur_data_lst):
            obj.set_data(data[[0,1],:])
            obj.set_3d_properties(data[2,:])

    
    ani_obj = animation.FuncAnimation(fig, update_links, body_pos.shape[0]-1, fargs=(link_data, link_obj),
                            interval=30, blit=False, repeat=True)
    
    return ax, ani_obj
