"""Point-cloud rendering for the demo video (a simple z-buffered splatter, no GPU or window needed)."""
import numpy as np


def render_cloud(xyz, rgb, yaw, size=(1088, 1080), pivot=(1.0, -0.3, 5.0), shift_y=0.8, pull_back=3.0, focal=1050.0, splat=4):
    """Render a coloured cloud from a virtual camera orbiting `pivot` by `yaw` radians. Returns a BGR image.

    xyz      (N, 3) points in the camera frame of the stereo rig (x right, y down, z forward), metres
    rgb      (N, 3) colours in 0..1
    shift_y  moves the cloud down on screen (metres); pull_back moves the virtual camera away from the scene
    splat    each point is drawn as a splat x splat square of pixels
    """
    w, h = size
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pivot = np.asarray(pivot, float)
    p = (xyz - pivot) @ R.T + pivot
    p[:, 1] += shift_y
    p[:, 2] += pull_back
    keep = p[:, 2] > 0.3
    p, rgb = p[keep], rgb[keep]
    u = (focal * p[:, 0] / p[:, 2] + w / 2).astype(int)
    v = (focal * p[:, 1] / p[:, 2] + h * 0.5).astype(int)
    order = np.argsort(-p[:, 2])                      # far to near, so near points overwrite far ones
    u, v, rgb = u[order], v[order], rgb[order]
    img = np.full((h, w, 3), 18, np.uint8)
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[m], uu[m]] = (rgb[m][:, ::-1] * 255).astype(np.uint8)    # points carry RGB, the canvas is BGR
    return img
