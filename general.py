import os
import PIL
from PIL import Image
import cv2
import numpy as np 
import torch
import torch.nn as nn 
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


ROOT = os.getcwd()
dataset_train = datasets.ImageFolder(ROOT+'/data/face/')
class_train = {i:c for c,i in dataset_train.class_to_idx.items()}


def change_type(face, PIL=False):
	''' Chuyển dạng dữ liệu về dạng tensor
	'''
	if PIL:
		face_PIL = face
	else:
		face_cv2 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Chuyển về ảnh RGB
		face_PIL = Image.fromarray(face_cv2) # Chuyển về định dang PIL

	face_resize = face_PIL.resize((160, 160), Image.BILINEAR) # Resize images
	face_tensor = transforms.functional.to_tensor(np.float32(face_resize)) # Change tensor
	processed_tensor = (face_tensor - 127.5) / 128.0
	face = processed_tensor.unsqueeze(0)
	return face



def load_model(last_layer='all_KNN'):
	resnet = InceptionResnetV1(pretrained='vggface2').eval()
	if last_layer == 'all_KNN':
		data_not_mask = torch.load(ROOT + '/models/parameter/embedding/103_classes_not_mask.pt')
		data_mask  = torch.load(ROOT + '/models/parameter/embedding/103_classes_mask.pt')
		return resnet, data_not_mask, data_mask 

	elif last_layer == 'fully':
		net_not_mask = Net()
		net_mask = Net()	

		checkpoint_not_mask = torch.load(ROOT+'/models/parameter/classify/model_not_mask.pt', map_location='cpu')
		net_not_mask.load_state_dict(checkpoint_not_mask['model_state_dict'])

		checkpoint_mask = torch.load(ROOT+'/models/parameter/classify/model_mask.pt', map_location='cpu')
		net_mask.load_state_dict(checkpoint_mask['model_state_dict'])

		return resnet, net_not_mask, net_mask

	elif last_layer == 'linear':
		resnet_01 = InceptionResnetV1(pretrained='vggface2',
                               		classify=True,
                               		num_classes=103,
                              		).eval()

		return resnet_01


def data_loader(path_loader) -> dict:
	''' Tạo biến kiểu dữ liệu dict để chứa tên người, label và tên ảnh từng dữ liệu của người đó
	 Return data = {names_person : [label , [name_image1, name_image2, ...]]}
	'''
	path_data, label = {}, 0

	os.chdir(ROOT+ path_loader)
	names_person = os.listdir() 
	
	for name in names_person:
		os.chdir(ROOT+ path_loader + '/' + name)
		all_name_images = os.listdir() 
		path_data[name] = [label, all_name_images]
		label += 1
	os.chdir(ROOT)

	return path_data



def all_KNN(img_crop, model, emb_face, threshold=0.5):
	''' Nhận ảnh khuôn mặt đã được cắt và trả về nhận diện đối tượng và khoảng cách 
		từ ảnh tới các điểm dữ liệu trung bình
		Var -img_crop: ảnh khuôn mặt đã được cắt type = np.array
			-threshold: đặt ngưỡng nếu khoảng cách quá xa thì là class ko xác định
	'''
	embeddings_list = [] # Chứa tất cả khoảng cách của face tới với toàn bộ dữ liệu
	names_list = [] # Chứa tên tất cả các đối tượng
	face = change_type(img_crop)

	output = model(face)

	for index in range(len(emb_face[0])):
		euclidean_distance = F.pairwise_distance(output, emb_face[0][index])
		embeddings_list.append(euclidean_distance)
		names_list.append(emb_face[1][index])

	min_list = min(embeddings_list)
	if min_list > threshold:
		return 'No_name', min_list.double()
	else:
		return names_list[embeddings_list.index(min_list)], min_list.double()


def liner_InceptionResnetv1(img_crop, resnet, class_train, threshold=3.5):
	face = change_type(img_crop) 

	output = resnet(face)	

	pred = output.data.max(1, keepdim=True)[1]
	print(class_train[pred[0].item()], output[0][pred[0].item()])
	if output[0][pred[0].item()] >= threshold:
		return class_train[pred[0].item()]
	else:
		return "No Name"


class Net(nn.Module):
    def __init__(self, num_classes=103):
        super(Net, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x) 
        return x



def classify_FC(img_crop, resnet, net, class_train, threshold=3.5):
    face = change_type(img_crop)   
        
    #Embbedding
    emb = resnet(face)
        
    #Classify
    output = net(emb)
    
    #Softmax
    pred = output.data.max(1, keepdim=True)[1]
    print(class_train[pred[0].item()], output[0][pred[0].item()])
    if output[0][pred[0].item()] >= threshold:
        return class_train[pred[0].item()]
    else:
        return "No Name"


if __name__ == '__main__1':
	correct, total = 0, 0
	resnet, emb_not_mask, emb_mask = load_model(last_layer='all_KNN')

	arr_folder = os.listdir(ROOT+'/Test')

	for name_folder in arr_folder:
		print(name_folder)
		arr_file = os.listdir(ROOT+'/Test/'+name_folder)
		
		for name_file in arr_file:
			stream = open(ROOT+'/Test/'+name_folder+'/'+name_file, "rb")
			bytes = bytearray(stream.read())
			numpyarray = np.asarray(bytes, dtype=np.uint8)
			im = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

			result, distance = all_KNN(im, model=resnet, emb_face=emb_not_mask, threshold=0.6)

			if result == name_folder:
				correct += 1

			elif result == 'No_name':
				print('Folder: {}, name of file: {}, predict {}, distance: {}'.format(name_folder, name_file, result, distance))
				print()

			else:
				print('Folder: {}, name of file: {}'.format(name_folder, name_file))
				print('Label: {}, predict: {}, distance: {}'.format(name_folder, result, distance))
				print()

		total += len(arr_file)
		print()
		print()

	print(f'Accuracy: {correct}/{total},  {(correct/total)*100} %')

if __name__ == '__main__':
	resnet, net_not_mask, net_mask = load_model(last_layer='fully')