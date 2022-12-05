import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from facenet_pytorch import InceptionResnetV1

import images
from general import data_loader, load_model, change_type

ROOT = os.getcwd() # Đường dẫn của file


def save_crop_face(path=ROOT+'/runs/crops') -> None:
	''' Xóa thư mục ảnh và crop từng khuôn mặt trên ảnh trong Person_data
		và lưu trữ tại runs/crops/
	'''
	path_data = data_loader('/data/face') # Load thư mục 

	# Delete folders data trong runs/crops/
	os.chdir(path)
	folders = os.listdir()
	for folder in folders:
		os.chdir(path + '/' + folder)
		files = os.listdir()

		for file in files:
			os.remove(file)
		os.chdir(path)
		os.rmdir(folder)
	os.chdir(ROOT)

	# Crop face bằng yolo trong file detect_image
	for name in path_data.keys():
		images.run(
			weights=ROOT + '/models/parameter/YOLOv5/weight_train.pt',
			source=ROOT + '/data/face/' + name +'/', # file/dir/URL/glob, 0 for webcam
			data=ROOT + '/data/facemask.yaml',
			imgsz=(640, 640),  # inference size (height, width)
			conf_thres=0.25,  # confidence threshold
			iou_thres=0.45,  # NMS IOU threshold
			max_det=1000,
			save_crop=True,  # save cropped prediction boxes
			view_img=False,  # maximum detections per image
			project=ROOT + '/runs/crops',  # save results to project/name
			name=name, # Name of persons
			)



def embedding_data():
	''' Save feature khuôn mặt crop đã trích đặc trưng 
	'''
	resnet = InceptionResnetV1(pretrained='vggface2').eval()

	dataset = datasets.ImageFolder('runs/crops') 
	idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} 

	name_list = [] 
	embedding_list = [] 

	for img_crop, idx in dataset:
		face = change_type(img_crop, PIL=True)    

		emb = resnet(face)

		embedding_list.append(emb.detach()) 
		name_list.append(idx)#idx_to_class[idx])

	data = [embedding_list, name_list]
	torch.save(data, ROOT + '/models/parameter/embedding/SVM_mask_test.pt') 



def search_center():
	''' Tìm tất cả các khoảng cách và vector trung bình của từng class
	'''
	data = torch.load(ROOT + '/models/parameter/embedding/vggface2.pt')
	embedding_list = data[0]
	name_list = data[1]

	average_knn = {}
	face_list = {name: [] for name in set(name_list)}

	for idx in range(len(embedding_list)):
		face_list[name_list[idx]].append(embedding_list[idx]) # Data đã được trích xuất đặc trưng
	
	for name in set(name_list):
		faces_tensor = torch.cat(face_list[name], dim=0) # Change to tensor
		average_knn[name] = torch.sum(faces_tensor, dim=0).div(len(face_list[name]))

	torch.save(average_knn, ROOT + '/models/parameter/KNN/vggface2.pth')


if __name__ == '__main__':
	save_crop_face()
	# embedding_data()
	#search_center()