import smplx
import json
import torch
import numpy as np
import os.path as op
import pickle as pkl
import render_utils
from PIL import Image
from glob import glob
from tqdm import tqdm


MANO_DIR_L = "./data/meta_data/model/MANO_LEFT.pkl"
MANO_DIR_R = "./data/meta_data/model/MANO_RIGHT.pkl"
DEVICE = "cuda"
ROOT_PATH = "./data/InterHand2.6M"


def constructor(capture_idx, seq_name, cam_idx, split):
    render_utils.mkdir('./outputs')
    # mano layer
    reg_path = "./data/meta_data/J_regressor_mano_ih26m.npy"
    ih26m_joint_regressor = np.load(reg_path)
    
    mano_layer = {
        "right": smplx.create(
            model_path=MANO_DIR_R, model_type="mano", use_pca=False, is_rhand=True
        ),
        "left": smplx.create(
            model_path=MANO_DIR_L, model_type="mano", use_pca=False, is_rhand=False
        ),
    }

    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if (
        torch.sum(
            torch.abs(
                mano_layer["left"].shapedirs[:, 0, :]
                - mano_layer["right"].shapedirs[:, 0, :]
            )
        )
        < 1
    ):
        mano_layer["left"].shapedirs[:, 0, :] *= -1

    root_path = ROOT_PATH
    img_root_path = op.join(root_path, "images")  # path to interhand images
    annot_root_path = op.join(root_path, "annotations")
    mano_p = "InterHand2.6M_%s_MANO_NeuralAnnot.json" % (split)
    cam_p = "InterHand2.6M_%s_camera.json" % (split)
    joints_p = "InterHand2.6M_%s_joint_3d.json" % (split)

    with open(op.join(annot_root_path, split, mano_p)) as f:
        mano_params = json.load(f)
    with open(op.join(annot_root_path, split, cam_p)) as f:
        cam_params = json.load(f)
    with open(op.join(annot_root_path, split, joints_p)) as f:
        joints = json.load(f)

    img_path_list = glob(
        op.join(
            img_root_path,
            split,
            "Capture" + capture_idx,
            seq_name,
            "cam" + cam_idx,
            "*.jpg",
        )
    )

    for k, layer in mano_layer.items():
        layer.cuda()

    ih26m_joint_regressor = torch.FloatTensor(ih26m_joint_regressor).unsqueeze(0).cuda()

    # MANO faces with part segmentation but water tight
    sealed_faces = np.load("./data/meta_data/sealed_faces.npy", allow_pickle=True).item()
    faces = sealed_faces["sealed_faces_right"]
    faces_color = sealed_faces["sealed_faces_color_right"]

    mano_faces_r = torch.LongTensor(faces).to(DEVICE)
    mano_faces_l = mano_faces_r[:, np.array([1, 0, 2])]  # opposite face normal
    mano_faces_l += render_utils.N_VERTEX
    mano_faces = torch.cat((mano_faces_r, mano_faces_l), dim=0).unsqueeze(0)

    part_texture = render_utils.get_part_texture(
        mano_faces[0].detach().cpu().numpy(),
        n_vertices=render_utils.N_VERTEX * 2,
        face2label=faces_color,
    ).to(DEVICE)

    dummpy_vertices = np.zeros((render_utils.N_VERTEX, 3)) + -1e20

    objs = {}
    objs["ih26m_joint_regressor"] = ih26m_joint_regressor
    objs["mano_params"] = mano_params
    objs["cam_params"] = cam_params
    objs["mano_faces"] = mano_faces
    objs["joints"] = joints
    objs["img_path_list"] = img_path_list
    objs["mano_layer"] = mano_layer
    objs["part_texture"] = part_texture
    objs["dummpy_vertices"] = dummpy_vertices
    return objs


def main(split, capture_idx, seq_name, cam_idx):
    # prepare objects for rendering
    objs = constructor(capture_idx, seq_name, cam_idx, split)
    mano_params = objs["mano_params"]
    cam_params = objs["cam_params"]
    mano_layer = objs["mano_layer"]
    joints = objs["joints"]
    pbar = tqdm(objs["img_path_list"])

    meta_dict = {}
    for img_path in pbar:
        img_key = "/".join(img_path.split("/")[-5:])
        meta_dict[img_key] = {}
        split, capture_idx, seq_name, cam_idx, _ = img_path.split("/")[-5:]
        capture_idx = capture_idx.replace("Capture", "")
        cam_idx = cam_idx.replace("cam", "")

        folder_path = op.dirname(
            img_path.replace("./data/InterHand2.6M/images", "./outputs/segms")
        )

        # out folder and path
        render_utils.mkdir(folder_path)
        out_path = op.join(folder_path, op.basename(img_path).replace(".jpg", ".png"))

        frame_idx = op.basename(img_path)[5:-4]
        out_dict = {}
        for hand_type in ("right", "left"):
            meta_dict[img_key][hand_type + "_fit_err"] = None
            # get mesh coordinate
            try:
                mano_param = mano_params[capture_idx][frame_idx][hand_type]
                if mano_param is None:
                    continue
            except KeyError:
                continue
            out_dict[hand_type] = {}

            # get MANO 3D mesh coordinates (world coordinate)
            mano_pose = torch.FloatTensor(mano_param["pose"]).view(-1, 3).to(DEVICE)
            root_pose = mano_pose[0].view(1, 3)
            hand_pose = mano_pose[1:, :].view(1, -1)
            shape = torch.FloatTensor(mano_param["shape"]).view(1, -1).to(DEVICE)
            trans = torch.FloatTensor(mano_param["trans"]).view(1, -1).to(DEVICE)
            output = mano_layer[hand_type](
                global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans
            )

            # milimeter
            mesh = (output.vertices[0] * 1000).cpu().detach().numpy()
            mesh = render_utils.add_seal_vertex(mesh)  # make MANO watertight

            # apply camera extrinsics
            cam_param = cam_params[capture_idx]
            t = np.array(cam_param["campos"][str(cam_idx)], dtype=np.float32).reshape(3)
            R = np.array(cam_param["camrot"][str(cam_idx)], dtype=np.float32).reshape(
                3, 3
            )
            t = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
            t = torch.FloatTensor(t).view(1, 1, 3).to(DEVICE)
            R = torch.FloatTensor(R).view(1, 3, 3).permute(0, 2, 1).to(DEVICE)
            mesh = torch.FloatTensor(mesh).view(1, -1, 3).to(DEVICE)
            mesh = torch.bmm(mesh, R) + t

            out_dict["im_path"] = img_path
            out_dict[hand_type]["mesh_cam"] = mesh.cpu().detach().numpy()

            out_dict["focal"] = np.array(
                cam_param["focal"][cam_idx], dtype=np.float32
            ).reshape(2)
            out_dict["princpt"] = np.array(
                cam_param["princpt"][cam_idx], dtype=np.float32
            ).reshape(2)

            # fitting error
            # This mesh is in camera coordinate now
            fit_err = render_utils.get_fitting_error(
                mesh[0][:-1],
                objs["ih26m_joint_regressor"],
                cam_params,
                joints,
                hand_type,
                capture_idx,
                frame_idx,
                cam_idx,
            )
            pbar.set_description("Fitting error: " + str(fit_err) + " mm")
            meta_dict[img_key][hand_type + "_fit_err"] = fit_err

        if "im_path" not in out_dict.keys():
            print("Do not have MANO; skip: " + img_path)
            continue

        im_path = out_dict["im_path"]
        focal = out_dict["focal"]
        princpt = out_dict["princpt"]

        # quick-and-dirty implementation
        # dummpy hand that will not be rendered because it is far away
        if "left" in out_dict.keys():
            mesh_cam_l = out_dict["left"]["mesh_cam"]
        else:
            mesh_cam_l = objs["dummpy_vertices"]

        if "right" in out_dict.keys():
            mesh_cam_r = out_dict["right"]["mesh_cam"]
        else:
            mesh_cam_r = objs["dummpy_vertices"]

        mesh_cam_l = mesh_cam_l.reshape(1, -1, 3)
        mesh_cam_r = mesh_cam_r.reshape(1, -1, 3)

        # image dimensions are needed for rendering
        im = Image.open(im_path)
        im_size = im.size
        im_w, im_h = im_size

        rend_dict = render_utils.render_mask(
            focal,
            princpt,
            mesh_cam_l,
            mesh_cam_r,
            im_size,
            objs["mano_faces"],
            objs["part_texture"],
            DEVICE,
        )
        parts = rend_dict["parts"][0].astype(np.uint8)
        parts = parts[:im_h, :im_w]

        parts_im = Image.fromarray(parts.astype(np.uint8))
        parts_im.save(out_path)
        meta_dict[img_key]["imsize"] = im.size
        im_arr = np.array(Image.open(out_path), dtype=np.uint8)
        assert np.abs(im_arr - parts).sum() == 0
    with open("./outputs/meta_dict_%s.pkl" % (split), "wb") as f:
        pkl.dump(meta_dict, f)


if __name__ == "__main__":
    split = "train"
    split = "val"

    capture_idx = "*"
    seq_name = "*"
    cam_idx = "*"
    main(split, capture_idx, seq_name, cam_idx)
