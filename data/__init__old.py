from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .dtu_ft_new import DTU_ft_new

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft_new}