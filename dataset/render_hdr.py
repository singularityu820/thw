import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import mitsuba as mi
import numpy as np
from tqdm import tqdm


def load_svbrdf(path):
    svbrdf = cv2.imread(path)[:, 512:]
    normal, diffuse, roughness, specular = np.split(svbrdf[:, :, ::-1], 4, 1)

    normal = normal / 255.0
    diffuse = diffuse / 255.0
    roughness = roughness / 255.0
    specular = specular / 255.0

    return normal, diffuse, roughness, specular, svbrdf


mi.set_variant("cuda_ad_rgb")

scene = mi.load_file("dataset/resource/hdr.xml")

params = mi.traverse(scene)

svbrdf_root = "/home/xiaojiu/datasets/MatSynth/test_linear"
svbrdf_names = os.listdir(svbrdf_root)
svbrdf_names.sort()

normal = params["OBJMesh.bsdf.normalmap.data"].numpy()
diffuse = params["OBJMesh.bsdf.nested_bsdf.bsdf_0.reflectance.data"].numpy()
roughness = params["OBJMesh.bsdf.nested_bsdf.bsdf_1.alpha.data"].numpy()
specular = params["OBJMesh.bsdf.nested_bsdf.bsdf_1.specular_reflectance.data"].numpy()

for svbrdf_name in tqdm(svbrdf_names):
    svbrdf_path = os.path.join(svbrdf_root, svbrdf_name)
    normal, diffuse, roughness, specular, svbrdf = load_svbrdf(svbrdf_path)

    params["OBJMesh.bsdf.nested_bsdf.bsdf_0.reflectance.data"] = diffuse
    params["OBJMesh.bsdf.nested_bsdf.bsdf_1.alpha.data"] = roughness**2
    params["OBJMesh.bsdf.nested_bsdf.bsdf_1.specular_reflectance.data"] = specular
    params["OBJMesh.bsdf.normalmap.data"] = normal
    params.update()

    img = mi.render(scene, params=params)

    mi.Bitmap(img).write(
        os.path.join("/home/zjj/dataset/MatSynth/test_HDR_Linear", f"{svbrdf_name}.exr")
    )

    cv2.imwrite(
        os.path.join("/home/zjj/dataset/MatSynth/test_HDR_Linear", f"{svbrdf_name}.png"), svbrdf
    )
