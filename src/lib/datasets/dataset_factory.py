from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Just for detection support coco and voc """
"""I recommend the voc because you could mark the boxes by LabeImg to get the voc conveniently """
""" Anyway, if you like, you can also use  convert the voc to coco and use CTDetDatasetCOCO """
from .ctdet_cocoface import CTDetFaceCOCODataset
from .ctdet_vocface import CTDetFaceVOCDataset
from .ctdet_voc import CTDetDatasetVOC
from .ctdet_coco import CTDetDatasetCOCO

"""face landmark and face box  similar to original pose estimation"""
"""refered to : https://github.com/bleakie/CenterMulti  """
from .multi_pose_face import MultiPoseDataset

"""face landmark regression instead of heatmap way"""
from .ctdet_vockptsreg import CTDetFacekptsDataset

dataset_factory = {
  'voc_facedet': CTDetFaceVOCDataset,
  'voc_det':CTDetDatasetVOC,
  'coco_det': CTDetDatasetCOCO,
  'coco_facedet': CTDetFaceCOCODataset,
  'coco_facedet_hplm': MultiPoseDataset,
  'voc_facedet_reglm':CTDetFacekptsDataset,
}

def get_dataset(dataset,task):
  # class Dataset(dataset_factory[dataset]):
  #   pass
  Dataset=dataset_factory[dataset]
  return Dataset
  
