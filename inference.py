import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import time
if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar
from deephar.config import ModelConfig
from deephar.config import pennaction_dataconf
from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

import argparse

act_list = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl', 'clean_and_jerk',
        'golf_swing', 'jump_rope', 'jumping_jacks', 'pullup', 'pushup', 'situp', 'squat',
        'strum_guitar', 'tennis_forehand', 'tennis_serve']

def parse_arguments():

	parser = argparse.ArgumentParser()
	parser.add_argument("--threshold", type = int, help= "Confidence threshold required to mark a seq as a specific class", default = 2)
	parser.add_argument("--seq_len", type = int, help = "Number of frames per seq", default = 8)
	#------------------------Output Files---------------------
	parser.add_argument("--per_frame_csv", type = str, help = "Name of csv file storing per frame prediction", default = "PerFramePrediction.csv" )
	parser.add_argument("--per_seq_csv", type = str, help = "Name of csv file storing per seq prediction", default = "Action_Prediction.csv" )
	parser.add_argument("--output_vid", type = str, help = "Name of Output video file", default = "Output.mp4")
	#------------------------Input Files---------------------
	parser.add_argument("--input_vid", type = str, help = "Name of Input video file", default = "Action1.mp4")
	#------------------------Actions---------------------
	parser.add_argument("--pred_pose", help = "Flag for predicting pose", action = 'store_true')
	parser.add_argument("--draw_pose", help = "Set this and pred_pose flag true to draw pose on frames", action = 'store_true')
	parser.add_argument("--store_results", help = "For Storing results as in an excel", action = 'store_true')
	
	args = parser.parse_args()
	return args

def buildModel(args):
	start = time.time()
	num_frames = args.seq_len
	cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
	                  num_actions=[15], num_pyramids=6, action_pyramids=[5, 6],
	                  num_levels=4, pose_replica=True,
	                  num_pose_features=160, num_visual_features=160)

	num_predictions = spnet.get_num_predictions(
	    cfg.num_pyramids, cfg.num_levels)
	num_action_predictions = \
	    spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

	full_model = spnet.build(cfg)

	weights_file = 'weights/weights_mpii+penn_ar_028.hdf5'

	full_model.load_weights(weights_file, by_name=True)

	models = split_model(full_model, cfg, interlaced=False,
	                     model_names=['2DPose', '2DAction'])
	end = time.time()
	print("Time Taken to build model : ", end - start)
	return models[0], models[1]


def extract_frames(filePath):
    frame = np.zeros((8, 256, 256, 3))
    fps = 0
    vc = cv.VideoCapture(filePath)
    num_frames = int(vc.get(cv.CAP_PROP_FRAME_COUNT))

    (major_ver, _, _) = (cv.__version__).split('.')
    if int(major_ver) < 3:
        fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vc.get(cv.CAP_PROP_FPS)
     
    ls = []
    i = 0

    img_dir='split/'
    set_number = 1
    succ, fr = vc.read()
    while succ:
        # print(i)
        if(i < 8):
            frame[i] = cv.resize(fr, (256, 256), cv.INTER_AREA)
            frame[i] = cv.cvtColor(frame[i].astype(
                np.float32), cv.COLOR_BGR2RGB)
            i = i + 1
        else:
            ls.append(frame)
            for k in range(len(frame)):
            	cv.imwrite(img_dir+str(set_number)+"/"+str(k)+".jpg", frame[k])
            set_number +=1	
            frame = np.concatenate((frame[1:], np.expand_dims(cv.cvtColor(cv.resize(
                fr, (256, 256), cv.INTER_AREA).astype(np.float32), cv.COLOR_BGR2RGB), axis=0)), axis=0)
        succ, fr = vc.read()
    ls.append(frame)
    batch = np.asarray(ls)/255.0
    return batch, fps, num_frames


def extract_frames_from_folder(folder_path):

    frames = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        if img is not None:
            # frames.append(cv.cvtColor(cv.resize(img,(256,256),cv.INTER_AREA).astype(np.float32), cv.COLOR_BGR2RGB))
            frames.append(np.array(img.resize((256, 256), Image.BILINEAR)))
            #print(np.array(img.resize((256, 256), Image.BILINEAR)))
    ls = []
    for i in range(len(frames)-7):
        ls.append(frames[i:i+8])

    batch = np.asarray(ls)/255.0

    return batch


def draw_single_frame(frame, pose):
    out_frame = frame
    color_tuple = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                   (255, 255, 0), (255, 0, 255)]
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
             [10, 12], [12, 14], [11, 13], [13, 15]]
    rad = 1
    # Draw joints
    for i in range(16):
        out_frame = cv.circle(out_frame, (int(pose[i][0]), int(
            pose[i][1])), rad, color_tuple[cmap[i]], -1)
    # Draw lines
    for l in links:
        out_frame = cv.line(out_frame, (int(pose[l[0]][0]), int(pose[l[0]][1])), (int(
            pose[l[1]][0]), int(pose[l[1]][1])), color_tuple[cmap[l[0]]], thickness=2)
    #font = cv.FONT_HERSHEY_SIMPLEX
    #out_frame = cv.putText(out_frame, act_list[act], (10, 20), font, 0.5, (255,255,255), 2, cv.LINE_AA)

    return out_frame


def draw_pose_on_frames(frames, poses, fps):
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (frames.shape[1], frames.shape[2])
    video = cv.VideoWriter('Output1.mp4', fourcc, float(fps), size)
    i = 0
    for frame, pose in zip(frames, poses) :
        if(i < 8):
            video.write(cv.normalize(draw_single_frame(frame, pose),
                                    None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            i = i + 1
        else :
            if(i%8 == 7):
                video.write(cv.normalize(draw_single_frame(frame, pose),
                                        None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
            i = i + 1

    video.release()


def get_prediction(pose_model, action_model, frames, args):

	print("Number of frame sequences: ", len(frames))
	start = time.time()
	pose = None

	action_list = action_model.predict(frames)
	if args.pred_pose:
		pose_list = pose_model.predict(frames)
		pose, action = pose_list[-1], action_list[-1]
		pose = np.resize(pose, (pose.shape[0]*pose.shape[1], pose.shape[2], pose.shape[3]))
	else: 
		action = action_list[-1]

	act = np.argmax(action, axis=1)
	end = time.time()

	print("time taken to get predictions : ", end - start)

	distribution = {'Frame set': range(1, len(action)+1)}
	for i in range(len(act_list)):
		distribution[act_list[i]]= np.array(action)[:,i]

	pd.DataFrame(distribution).to_csv('Distribution.csv', index=False)


	return pose, act


def store_action_prediction(act_idx):
    # act_list = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl', 'clean_and_jerk',
    #             'golf_swing', 'jump_rope', 'jumping_jacks', 'pullup', 'pushup', 'situp', 'squat',
    #             'strum_guitar', 'tennis_forehand', 'tennis_serve']
    act_pred = [act_list[i] for i in act_idx]
    pd.DataFrame({'Frame set': range(1, len(act_pred)+1),
                  'Action_Label': act_pred}).to_csv('Action_Prediction.csv', index=False)


def get_per_frame_prediction(act_res, num_frames, args):

	start = time.time()
	act_pred = [act_list[i] for i in act_res]

	
	per_seq_pred_thresh = [-1]*len(act_pred)

	seq_len = args.seq_len
	threshold = args.threshold
	band = 2*threshold + 1

	index = 0

	for i in range(len(act_pred) - threshold):
		temp_array = act_pred[i : i+band]
		flag = True
		for j in range(len(temp_array)):
			if temp_array[0] == temp_array[j]:
				continue
			else:
				flag = False

		if flag :
			per_seq_pred_thresh[i+threshold] = temp_array[0]

	per_frame_pred = [-1]*num_frames		

	for i in range(num_frames):

		if i < seq_len//2 -1:
		# per_frame_pred[i] = -1
			continue

		elif i >= num_frames - seq_len//2:
		# per_frame_pred[i] = -1	
			continue
		else:
			per_frame_pred[i] = per_seq_pred_thresh[i - seq_len//2 + 1]

	end = time.time()
	print("time taken to get per frame prediction", end - start)

	#adding null values so that column can be added to excel
	act_pred.extend([-1]*7)
	per_seq_pred_thresh.extend([-1]*7)

	if args.store_results:
		pd.DataFrame({'Frame': range(1, len(per_frame_pred)+1),
		'Per_Frame_Action_Label': per_frame_pred,
		'Per_Seq' : act_pred,
		'Per_Seq_Thresh ' : per_seq_pred_thresh}).to_csv(args.per_frame_csv, index=False)

	# print(num_frames)

def main():
	start = time.time()
	args = parse_arguments()
	pose_model, action_model = buildModel(args)
	fPath = args.input_vid
	inp, fps, num_frames = extract_frames(fPath)
	#inp = extract_frames_from_folder('datasets/PennAction/frames/0946')
	pose_res, act_res = get_prediction(pose_model, action_model,inp, args )
	
	if args.draw_pose:
		frames = 255.0*np.resize(inp, (inp.shape[0]*inp.shape[1], 256, 256, 3))
		draw_pose_on_frames(frames, 256*pose_res, fps)
	
	get_per_frame_prediction(act_res, num_frames, args)
	# store_action_prediction(act_res)
	end = time.time()
	print("Time taken for whole script to run : ", end - start)

if __name__ == '__main__':
    main()
