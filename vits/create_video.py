import json
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
from tensorflow.python.keras import Sequential 
from tensorflow.python.keras.models import load_model
import pickle
import torch
from nltk.stem.lancaster import LancasterStemmer
from BLIP import getContext
from BLIP.models.blip import blip_decoder
from TTS import create_context_response
from moviepy.editor import *
device = torch.device("cpu")

import json
from difflib import SequenceMatcher


answered_contexts = []

def answer_context(context):
    with open('../contextModelFiles/Contexts.json', 'r+') as openfile:
        contexts = json.load(openfile)
    
    for b in contexts['Contexts']:
        ratio = SequenceMatcher(None, context, b["context"]).ratio()
        #print("input: ", i["context"],"to test: ", b["context"], ratio)
        if ratio >= 0.90:
            for c in b["responses"]:
                if c not in answered_contexts:
                    answered_contexts.append(c)
                    return c
    return None


def create_video_order_text_file(length):
    for i in range(length):
        with open('order.txt', 'a') as f:
            f.write(('\nfile '+"'"+str(i)+".mp4"+"'"))
            f.close()

create_video_order_text_file(245)



def create_video(video_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(video_path)
    current_starting_time = 2
    video_length = video.duration
    video_finished = False

    image_size = 384
    model_path = './BLIP/models/model.pth'
    model = blip_decoder(pretrained=model_path, image_size=image_size, vit='large')
    model.eval()
    model = model.to(device) 

    current_clip_index = 1

    # add the first 2 secs of the video to clips
    intro_video_clip = video.subclip(0, 2).fx(afx.volumex, 0.5)
    intro_video_clip.write_videofile(("../clips/0.mp4"), codec="libx264", audio_codec="aac")


    while video_finished == False:
        #print(current_starting_time)
        if current_starting_time+1 >= video_length:
            video_finished == True
            break

        # gets and saves a frame for getting the context
        clip = video.subclip(current_starting_time, current_starting_time+1) 
        clip.save_frame("../image_to_get_context/1.png", t=2)


        Image_Context = getContext.get_image_context("../image_to_get_context/1.png", model)
        answer = answer_context(Image_Context)

        print("context: ", Image_Context, "answer: ", answer)

        if answer == "None" or answer == None:
            # no_context_video_clip = video.subclip(current_starting_time, (current_starting_time+1)).fx(afx.volumex, 0.5)
            # no_context_video_clip.write_videofile(("../clips/"+str(current_clip_index)+".mp4"), codec="libx264", audio_codec="aac")
            # print("Start time: ", current_starting_time, "End time: ", (current_starting_time+1))
            # current_clip_index += 1
            current_starting_time += 1
            print("no contexts found")
        else:
            create_context_response(answer, "./response.wav")
            audio_response = AudioFileClip("./response.wav")

            print("here, video length: ", video_length)
            if audio_response.duration+current_starting_time+1 < video_length:
                response_video_clip = video.subclip(current_starting_time, (audio_response.duration+current_starting_time+1))
                video_audio = audio.subclip(current_starting_time, (audio_response.duration+current_starting_time+1)).fx(afx.volumex, 0.5)
                
                print("Start time: ", current_starting_time, "End time: ", (audio_response.duration+current_starting_time+2))
                new_audioclip = CompositeAudioClip([video_audio, audio_response])
                
                response_video_clip.audio = new_audioclip

                response_video_clip.write_videofile(("../clips/"+str(current_clip_index)+".mp4"), codec="libx264", audio_codec="aac")
                
                # current_clip_index += 1
                # print("in between start: ", audio_response.duration, "end: ", response_video_clip.duration)
                # pause_video_clip = response_video_clip.subclip(audio_response.duration, response_video_clip.duration).fx(afx.volumex, 0.5)
                # pause_video_clip.write_videofile(("../clips/"+str(current_clip_index)+".mp4"), codec="libx264", audio_codec="aac")
                # print("Start time: ", audio_response.duration, "End time: ", response_video_clip.duration)


                current_starting_time += (audio_response.duration)
                current_clip_index += 1



    # audio = AudioFileClip("1.mp3")
    # video.audio = CompositeAudioClip([audio, video.subclip(1, 5).audio.set_start(1), audio.set_start(5), clip.subclip(6, clip.duration).audio.set_start(6)])
    # video.write_videofile('final_video.mp4', codec="libx264", audio_codec="aac")



create_video("../video/short.mp4")
