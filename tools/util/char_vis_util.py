from matplotlib.colors import ListedColormap
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

def vis_pose(pose, char_model, key_body_ids=None, plot_text=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_pose(fig, ax, pose, char_model, key_body_ids=key_body_ids, plot_text=plot_text)
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, 2.0)
    plt.show()
    return

def vis_compare_anim(src_body_pos, tgt_body_pos, 
                     src_char_model, tgt_char_model, 
                     speedup=1, plot_ground=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    src_parent_indices = src_char_model._parent_indices
    tgt_parent_indices = tgt_char_model._parent_indices

    speedup = int(speedup)
    ax, src_ani_obj = plot_ani(fig, ax, src_body_pos[::speedup], src_parent_indices, colors='r')
    ax, tgt_ani_obj = plot_ani(fig, ax, tgt_body_pos[::speedup], tgt_parent_indices, colors='b')

    if plot_ground:
        ax = plot_grid_ground(ax)

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 3.0)
    plt.show()
    return

def vis_compare_poses(src_pose, tgt_pose, 
                    src_char_model, tgt_char_model, 
                    src_key_body_ids, tgt_key_body_ids,
                    plot_ground=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax = plot_pose(fig, ax, src_pose, src_char_model, key_body_ids=src_key_body_ids, plot_text=True, color='r')
    ax = plot_pose(fig, ax, tgt_pose, tgt_char_model, key_body_ids=tgt_key_body_ids, plot_text=True, color='b')

    if plot_ground:
        ax = plot_grid_ground(ax)
    
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, 2.0)
    plt.show()
    return

def plot_pose(fig, ax, t_pose, char_model, key_body_ids=None, plot_text=True, color='r'):
    root_pos, root_rot_quat, joint_rot_quat = retagreter.extract_frame_data(t_pose, char_model)
    body_pos, body_rot = char_model.forward_kinematics(root_pos, root_rot_quat, joint_rot_quat)
    
    body_pos = body_pos.detach().cpu().numpy()
    num_jnt = body_pos.shape[0]
    parents = char_model._parent_indices

    for i in range(num_jnt):
        if parents[i] != -1:
            ax.plot([body_pos[i,0], body_pos[parents[i],0]], 
                    [body_pos[i,1], body_pos[parents[i],1]], 
                    [body_pos[i,2], body_pos[parents[i],2]], color=color)
    
    if key_body_ids is None:
        ax.scatter(body_pos[:,0], body_pos[:,1], body_pos[:,2], color=color, marker='o')
    
    else:
        jnt_cmap = ["red", "blue", "green", "grey", "cyan", "brown", "pink", "purple", 
                    "orange", "aqua", "fuchsia", "silver", "magenta", "lime", "olive", 
                    "teal", "navy", "maroon"]
        
        for i in range(key_body_ids.shape[0]):
            src_id = key_body_ids[i]
            ax.scatter(body_pos[src_id,0], body_pos[src_id,1], body_pos[src_id,2], 
                    color=jnt_cmap[::-1][i % len(jnt_cmap)], marker='o')
            
            if plot_text:
                ax.text(body_pos[src_id,0], body_pos[src_id,1], body_pos[src_id,2], str(i), 
                            color=jnt_cmap[::-1][i % len(jnt_cmap)])
    return ax


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


def plot_grid_ground(ax):
    x = np.linspace(-0.5, 0.5, 10)
    y = np.linspace(-0.5, 0.5, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    checker = ((np.floor(X) + np.floor(Y)) % 2).astype(int)
    ground_cmap = ListedColormap(['black', 'white'])
    
    ax.plot_surface(X, Y, Z, facecolors=ground_cmap(checker), shade=True, alpha=0.6)
    return ax
