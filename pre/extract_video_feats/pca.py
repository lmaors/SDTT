#coding=utf-8
import numpy as np
from sklearn.decomposition import PCA
import h5py
import numpy as np
hdfFile = h5py.File('./c3d_features/c3d_fc7_features_all.hdf5', 'r')

data=[]
fileframe={}
for i in hdfFile.keys():
    print(i)
    dataset = hdfFile.get(i).get('c3d_features')
    dataset=np.array(dataset)
    data.extend(dataset)
    fileframe[i] = len(dataset)
data =np.array(data)
print(data.shape)


# print(hdfFile.keys())
# dataset1 = hdfFile.get('CIQ-mnURg9E.mp4').get('c3d_features')
# print(dataset1)
# print(dataset1.shape)
# data = np.array(dataset1)
# print(data)
# # for i in dataset1:
# #     print(i)
# hdfFile.close()
pca = PCA(n_components=500)
pca.fit(data)
newX = pca.fit_transform(data)
frame_start = 0
f = h5py.File('pca_500.hdf5', 'w')
for video_name in hdfFile.keys():
    print(video_name)
    frame_end = frame_start + fileframe[video_name]
    newfeature = newX[frame_start:frame_end]
    frame_start = frame_end

    total_frames = hdfFile.get(video_name).get('total_frames')
    valid_frames= hdfFile.get(video_name).get('valid_frames')


    fgroup = f.create_group(video_name)
    fgroup.create_dataset('c3d_features', data=newfeature)
    fgroup.create_dataset('total_frames', data=np.array(total_frames))
    fgroup.create_dataset('valid_frames', data=np.array(valid_frames))
    # newX=pca.fit_transform(data)
    # PCA(copy=True, n_components=2, whiten=False
    # print(pca.explained_variance_ratio_)
    # print(newX.shape)
    print(newfeature.shape,fileframe[video_name])





