#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD
from SyncNetInstance import *
import warnings
import random

warnings.filterwarnings("ignore")

class Args:
  def __init__(self, videofile, data_dir):
    self.logs = "/home/seokje/data_verification.csv"
    self.initial_model = "data/syncnet_v2.model"
    self.batch_size = 64
    self.vshift = 15
    self.data_dir = os.path.join(os.getcwd(), data_dir)
    self.videofile = videofile
    self.reference = ""
    self.facedet_scale = 0.25
    self.crop_scale = 0.40
    self.min_track = 100
    self.frame_rate = 25
    self.num_failed_det = 25
    self.min_face_size = 100
    self.avi_dir = os.path.join(self.data_dir, "pyavi")
    self.tmp_dir = os.path.join(self.data_dir, "pytmp")
    self.work_dir = os.path.join(self.data_dir, "pywork")
    self.crop_dir = os.path.join(self.data_dir, "pycrop")
    self.frames_dir = os.path.join(self.data_dir, "pyframes")

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})

  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(opt,track,cropfile):
  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
    
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

  # ========== CROP AUDIO FILE ==========

  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  # if output != 0:
  #   pdb.set_trace()

  # sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========

  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  # if output != 0:
  #   pdb.set_trace()

  # print('Written %s'%cropfile)

  os.remove(cropfile+'t.avi')

  # print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt, DET):
  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()

  dets = []
      
  for fidx, fname in enumerate(flist):

    start_time = time.time()
    
    image = cv2.imread(fname)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

    # dets.append([]);
    for bbox in bboxes:
      dets.append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
      # dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

    elapsed_time = time.time() - start_time

    # print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')
  if not dets:
    return None
  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)
  framenum = np.array([f["frame"] for f in dets])
  bboxes = np.array([f["bbox"] for f in dets])
  frame_i = np.arange(framenum[0], framenum[-1]+1)
  bboxes_i = []
  for ij in range(0, 4):
    interpfn = interp1d(framenum, bboxes[:, ij])
    bboxes_i.append(interpfn(frame_i))
  bboxes_i = np.stack(bboxes_i, axis=1)
  return [{"frame": frame_i, "bbox": bboxes_i}]
  # return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_manager = VideoManager([os.path.join(opt.avi_dir,opt.reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(opt.work_dir,opt.reference,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)

  # print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

  return scene_list
    

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========
def main(args):
  # define syncnet model
  s = SyncNetInstance()
  s.loadParameters("data/syncnet_v2.model")
  # define detector model
  DET = S3FD(device='cuda')

  i, video_path = args
  dir_path = os.path.join("/", *video_path.split("/")[:-1])
  listdir = os.listdir(dir_path)
  for path in listdir:
    if path.endswith(".csv"):
      csv_path = path
      break
  df = pd.read_csv(os.path.join(dir_path, csv_path), header=None)
  time_steps = [df[2].iloc()[i].item() for i, j in enumerate(df[1].iloc()) if j == "OK"]
  while True:
    interval = random.randint(5, 45)
    start_time = int((time_steps[interval+1] + time_steps[interval]) // 2)
    try:
      result = process(i, video_path, interval, start_time, s, DET)
      if result is None:
        continue
      else:
        return
    except:
      continue


def process(i, video_path, interval, start_time, s, DET):
  st = time.time()
  # i, (video_path, interval, start_time) = args
  pathes = video_path.split("/")
  video_dir = pathes[-2]
  video_name = pathes[-1].split(".")[0]
  output_path = video_dir + "_" + video_name + "_" + str(interval)
  print(i, output_path, "is being processed")
  opt = Args(video_path, os.path.join("processing", output_path))

  # set csv path per video
  current_path = os.path.dirname(opt.logs)
  video_validation_log_path = os.path.join(current_path, 'syncnet_results', output_path + '.csv')
  opt.video_validation_log = video_validation_log_path

  # ========== DELETE EXISTING DIRECTORIES ==========

  if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
    rmtree(os.path.join(opt.work_dir,opt.reference))

  if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
    rmtree(os.path.join(opt.crop_dir,opt.reference))

  if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
    rmtree(os.path.join(opt.avi_dir,opt.reference))

  if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
    rmtree(os.path.join(opt.frames_dir,opt.reference))

  if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
    rmtree(os.path.join(opt.tmp_dir,opt.reference))

  # ========== MAKE NEW DIRECTORIES ==========

  os.makedirs(os.path.join(opt.work_dir,opt.reference))
  os.makedirs(os.path.join(opt.crop_dir,opt.reference))
  os.makedirs(os.path.join(opt.avi_dir,opt.reference))
  os.makedirs(os.path.join(opt.frames_dir,opt.reference))
  os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

  # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========

  command = ("ffmpeg -y -ss %s -to %s  -i %s -vf 'transpose=2' -qscale:v 2 -async 1 -r 25 %s" % (start_time, start_time + 1, opt.videofile, os.path.join(opt.avi_dir, opt.reference, "video.avi")))
  output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.frames_dir,opt.reference,'%06d.jpg'))) 
  output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  command = ("ffmpeg -y -ss %s -to %s -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (start_time, start_time + 1, opt.videofile,os.path.join(opt.avi_dir,opt.reference,'audio.wav'))) 
  output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  # ========== FACE DETECTION ==========

  faces = inference_video(opt, DET)
  if not faces:
    return None

  # ========== FACE TRACKING ==========

  alltracks = [faces[0]]
  vidtracks = []

  # ========== FACE TRACK CROP ==========
  for ii, track in enumerate(alltracks):
    vidtracks.append(crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)))

  # ========== SAVE RESULTS ==========

  savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')

  flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
  flist.sort()

  # ==================== GET OFFSETS ====================

  dists = []
  if not flist:
    return None
  for idx, fname in enumerate(flist):
    offset, conf, dist, offset, minval, conf = s.evaluate(opt,videofile=fname)
    with open(opt.video_validation_log, "a") as f:
      f.write(f"{opt.videofile},{start_time-1},{offset},{minval:.3f},{conf:.3f},{time.time() - st:.3f}\n")
    dists.append(dist)
        
  # ==================== PRINT RESULTS TO FILE ====================

  torch.cuda.empty_cache()
  rmtree(opt.data_dir)
  return True


parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, required=True)
parser.add_argument("--logs", type=str, default="/home/seokje/data_verification.csv")
opt = parser.parse_args()

from pathlib import Path
import pandas as pd
from multiprocessing import Pool


if __name__ == "__main__":
    START_TIME = time.time()
    videos = [
        str(path)
        for path in Path(opt.data_root).rglob("*.mkv")
        if not path.name.startswith(".")]
    
    if not os.path.exists(opt.logs):
        with open(opt.logs, "w") as f:
            f.write("video,start_time,AV_offset,Min_dist,Confidence,Elapsed_time\n")
    logs = pd.read_csv(opt.logs)
    
    print("Total video count:", len(videos))
    videos_to_process = sorted(list(set(videos) - set(logs["video"])))
    print("Processing videos:", len(videos_to_process))
    print("-" * 100)

    num_workers = os.cpu_count()

    with Pool(num_workers) as pool:
        pool.map(main, enumerate(videos_to_process))
    print(time.time() - START_TIME)
