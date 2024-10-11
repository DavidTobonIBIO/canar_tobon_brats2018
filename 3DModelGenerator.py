import numpy as np
import nibabel as nib
import cv2
from skimage import measure
import trimesh
import matplotlib.pyplot as plt




def loadNiiFile(path):
    volume = nib.load(path).get_fdata()
    volume = volume.astype("uint8")

    return volume

def getBrainContours(image):
    rows, cols, floors = image.shape
    brainContours = np.zeros((rows, cols, floors), dtype="uint8")

    for i in range(floors):

        cannyImg = cv2.Canny(image[:, :, i], 100, 120)
        cannyImg = cv2.dilate(cannyImg, None)
        contoursList, hierarchy = cv2.findContours(cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contourImg = np.zeros((rows, cols))

        cv2.drawContours(contourImg, contoursList, -1, (255), 1)

        brainContours[:, :, i] = contourImg

    return brainContours == 255

def getTissueMasks(predictedVolume):
    tumorCore = predictedVolume == 1
    wholeTumor = predictedVolume == 2
    enhancingTumor = predictedVolume == 4

    return tumorCore, wholeTumor, enhancingTumor 

def get3DMesh(maskVolume, faceColors=np.array([0,0,0]), alpha=255):
    vertices, faces, normals, values = measure.marching_cubes(maskVolume)

    rows, cols = faces.shape
    colors = np.zeros((rows, 4))
    colors[:, :3] = faceColors 
    colors[:, 3] = alpha
    mesh = trimesh.Trimesh(vertices = vertices, faces=faces, face_colors=colors)
    return mesh


def savePredictionGLB(originalVolumePath, predictedVolumePath, glbFile):
    originalVolume = loadNiiFile(originalVolumePath)
    predictedVolume = loadNiiFile(predictedVolumePath)

    brainContours = getBrainContours(originalVolume)
    tumorCore, wholeTumor, enhancingTumor = getTissueMasks(predictedVolume)

    brainContourMesh = get3DMesh(brainContours, alpha=0.3)
    tumorCoreMesh = get3DMesh(tumorCore, np.array([0,246,246]), alpha=0.5)
    wholeTumorMesh = get3DMesh(wholeTumor, np.array([241,244,0]), alpha=0.4)
    enhancingTumorMesh = get3DMesh(enhancingTumor, np.array([255,0,0]), alpha=0.45)

    meshes = [enhancingTumorMesh,tumorCoreMesh, wholeTumorMesh, brainContourMesh, ]
    assembly = trimesh.util.concatenate(meshes)

    assembly.show()



def main():
    savePredictionGLB("Brats18_2013_4_1_t1.nii", 
                      "Brats18_2013_4_1_t1ce_seg.nii.gz", 
                      "pred3d.obj")


if __name__ == "__main__":
    main()