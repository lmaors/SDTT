# coding: utf-8
# from data_provider import *



import argparse
import os 

import pickle
import time
import numpy as np



def feature_extractor(BATCH_SIZE):
	#trainloader = Train_Data_Loader( VIDEO_DIR, resize_w=128, resize_h=171, crop_w = 112, crop_h = 112, nb_frames=16)
	
	# read video list from the txt list
	video_list_file = '2.txt'
	video_list = open(video_list_file).readlines()
	video_list = [item.strip() for item in video_list]
	print('video_list', video_list)


	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	

	# current location
	temp_path = os.path.join(os.getcwd(), 'temp')
	if not os.path.exists(temp_path):
		os.mkdir(temp_path)

	error_fid = open('error.txt', 'w')
	for video_name in video_list: 
		video_path = os.path.join(VIDEO_DIR, video_name)
		print('video_path', video_path)
		frame_path = os.path.join(temp_path, video_name)
		if not os.path.exists(frame_path):
			os.mkdir(frame_path)

		print('Extracting video frames ...')
		# using ffmpeg to extract video frames into a temporary folder
		# example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
		os.system('ffmpeg -i ' + video_path + ' -q:v 2 -f image2 ' + frame_path + '/image_%5d.jpg')
		print('Extracting features ...')




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print('******--------- Extract C3D features ------*******')
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./output_frm/', help='Output file name')
	parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=7, help='Feature extractor layer')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type = str, help='Input Video directory')
	parser.add_argument('-gpu', '--gpu', dest='GPU', action = 'store_true', help='Run GPU?')
	parser.add_argument('--OUTPUT_NAME', default='c3d_features.hdf5', help='The output name of the hdf5 features')
	parser.add_argument('-b', '--BATCH_SIZE', default=4, help='the batch size')
	parser.add_argument('-id', '--gpu_id', default=0, type=int)
	parser.add_argument('-p', '--video_list_file', type=str, help='the video name list')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print('parsed parameters:',params)

	OUTPUT_DIR = params['OUTPUT_DIR']
	EXTRACTED_LAYER = params['EXTRACTED_LAYER']
	VIDEO_DIR = params['VIDEO_DIR']
	RUN_GPU = params['GPU']
	OUTPUT_NAME = params['OUTPUT_NAME']
	BATCH_SIZE = params['BATCH_SIZE']
	crop_w = 112
	resize_w = 128
	crop_h = 112
	resize_h = 171
	nb_frames = 8
	feature_extractor(BATCH_SIZE)


