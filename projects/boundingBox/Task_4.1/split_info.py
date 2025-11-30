import numpy as np

data = np.load('/work3/s204164/DeepLearningCompVis/projects/boundingBox/Task_4.1/coord_proposals/dataset_crops_split/split_info.npy', allow_pickle=True)

# print(len(data.item(0)["train_images"]))
# print(len(data.item(0)["val_images"]))
print(data.item(0)["test_images"])