import torch


def index(feat, uv):
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
