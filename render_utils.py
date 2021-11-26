import numpy as np
import torch
import neural_renderer as nr
import os

N_PARTS = 16 * 2
N_VERTEX = 778 + 1


def add_seal_vertex(vertex):
    circle_v_id = np.array(
        [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
        dtype=np.int32,
    )
    center = (vertex[circle_v_id, :]).mean(0)
    vertex = np.vstack([vertex, center])
    return vertex


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_part_texture(faces, n_vertices, face2label):
    """
    :param faces: (numpy array Nx3)mesh faces numpy array
    :param n_vertices: (int) mesh number of vertices
    :return: texture (torch tensor 1xNx1x1x1x3) input to neural renderer
    """
    num_faces = faces.shape[0]
    half_faces = int(num_faces / 2)

    face2label = face2label[:, None]
    face2label = np.repeat(face2label, 3, axis=1)

    face_colors = np.ones((num_faces, 4))
    face_colors[:half_faces, :3] = face2label
    face_colors[half_faces:, :3] = face2label + 16

    texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
    texture[0, :, 0, 0, 0, :] = face_colors[:, :3] / N_PARTS
    texture = torch.from_numpy(texture).float()
    return texture


def generate_part_labels(
    vertices, faces, cam_t, neural_renderer, part_texture, K, R, part_bins
):
    """
    :param vertices: (torch tensor NVx3) mesh vertices
    :param faces: (torch tensor NFx3) mesh faces
    :param cam_t: (Nx3) camera translation
    :param neural_renderer: renderer
    :param part_texture: (torch tensor 1xNx1x1x1x3)
    :param K: (torch tensor 3x3) cam intrinsics
    :param R: (torch tensor 3x3) cam rotation
    :param part_bins: bins to discretize rendering into part labels
    :return: parts (torch tensor Bx3xWxH) part segmentation labels,
    :return: render_rgb (torch tensor Bx3xWxH) rendered RGB image
    """
    batch_size = vertices.shape[0]

    parts, depth, mask = neural_renderer(
        vertices,
        faces.expand(batch_size, -1, -1),
        textures=part_texture.expand(batch_size, -1, -1, -1, -1, -1),
        K=K.expand(batch_size, -1, -1),
        R=R.expand(batch_size, -1, -1),
        t=cam_t.unsqueeze(1),
    )

    render_rgb = parts.clone()

    parts = parts.permute(0, 2, 3, 1)
    parts *= 255.0  # multiply it with 255 to make labels distant
    parts, _ = parts.max(-1)  # reduce to single channel

    parts = torch.bucketize(parts.detach(), part_bins, right=True)
    parts = parts.long() + 1
    parts = parts * mask.detach()

    return parts.long(), render_rgb, depth.detach()


def render_mask(
    focal,
    princpt,
    mesh_cam_l,
    mesh_cam_r,
    im_size,
    mano_faces,
    part_texture,
    device,
):
    im_w, im_h = im_size
    imsize = max(im_size) + 10

    # initialize neural renderer
    # WARNING: always set directional light to 0 in order to avoid any shading in the rendered images
    neural_renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=imsize,
        image_size=imsize,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=False,
    ).cuda()

    scale = 1.0
    K = torch.FloatTensor(
        np.array(
            [[[focal[0], scale, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]]]
        )
    ).to(device)

    bins = (torch.arange(int(N_PARTS)) / float(N_PARTS) * 255.0) + 1
    bins = bins.to(device)

    # MANO is rotated 180 degrees in x axis, revert it.
    vertices_l = torch.FloatTensor(mesh_cam_l).to(device) / 1000
    vertices_r = torch.FloatTensor(mesh_cam_r).to(device) / 1000
    vertices = torch.cat((vertices_r, vertices_l), dim=1)
    R = torch.eye(3).to(device)
    cam_t = torch.zeros(1, 3).to(device)

    parts, render, depth = generate_part_labels(
        vertices=vertices,
        faces=mano_faces,
        cam_t=cam_t,
        K=K,
        R=R,
        part_texture=part_texture,
        neural_renderer=neural_renderer,
        part_bins=bins,
    )

    # below is needed for visualization only
    parts = parts.cpu().numpy()
    out_dict = {}
    out_dict["parts"] = parts
    out_dict["depth"] = depth
    out_dict["imsize"] = imsize
    return out_dict


def get_fitting_error(
    mesh, regressor, cam_params, joints, hand_type, capture_id, frame_idx, cam
):
    # ih26m joint coordinates from MANO mesh
    ih26m_joint_from_mesh = (
        torch.bmm(regressor, mesh.unsqueeze(0))[0].cpu().detach().numpy()
    )

    # camera extrinsic parameters
    t, R = np.array(
        cam_params[str(capture_id)]["campos"][str(cam)], dtype=np.float32
    ).reshape(3), np.array(
        cam_params[str(capture_id)]["camrot"][str(cam)], dtype=np.float32
    ).reshape(
        3, 3
    )
    t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t

    # ih26m joint coordinates (transform world coordinates to camera-centered coordinates)
    ih26m_joint_world = np.array(
        joints[str(capture_id)][str(frame_idx)]["world_coord"], dtype=np.float32
    ).reshape(-1, 3)
    ih26m_joint_cam = np.dot(R, ih26m_joint_world.transpose(1, 0)).transpose(
        1, 0
    ) + t.reshape(1, 3)
    ih26m_joint_valid = np.array(
        joints[str(capture_id)][str(frame_idx)]["joint_valid"], dtype=np.float32
    ).reshape(-1, 1)

    # choose one of right and left hands
    if hand_type == "right":
        ih26m_joint_cam = ih26m_joint_cam[np.arange(0, 21), :]
        ih26m_joint_valid = ih26m_joint_valid[np.arange(0, 21), :]
    else:
        ih26m_joint_cam = ih26m_joint_cam[np.arange(21, 21 * 2), :]
        ih26m_joint_valid = ih26m_joint_valid[np.arange(21, 21 * 2), :]

    # coordinate masking for error calculation
    ih26m_joint_from_mesh = ih26m_joint_from_mesh[
        np.tile(ih26m_joint_valid == 1, (1, 3))
    ].reshape(-1, 3)
    ih26m_joint_cam = ih26m_joint_cam[np.tile(ih26m_joint_valid == 1, (1, 3))].reshape(
        -1, 3
    )

    error = np.sqrt(np.sum((ih26m_joint_from_mesh - ih26m_joint_cam) ** 2, 1)).mean()
    return error
