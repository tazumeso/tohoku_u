#coding:utf-8

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import glob

def pitch2height(pitch, accidental):
	step = pitch[0].text
	octave = pitch[1].text
	pitch_dic = {"C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11}
	height = (int(octave) + 1) * 12 + pitch_dic[step] + accidental
	return height
	
def xml2list(file_name):
	tree = ET.parse(file_name)
	root = tree.getroot()
	part = root.find("part")
	mealist= part.findall("measure")
	attribute = part.find("measure").find("attributes")
	duration_len = int(attribute.find("divisions").text) * int(attribute.find("time").find("beats").text)
	dl = duration_len / 16.
	music_image = []
	highest = 0
	lowest = 88
	for mea in mealist:
		mea_image = [[0] * 40 for i in range(16)] 
		rhythm_image = [[0] * 40 for i in range(16)]  
		notelist = mea.findall("note")
		duration_point = 0
		for note in notelist:
			if note.find("pitch") is not None:
				accidental = 0
				if note.find("accidental") is not None:
					accidental = note.find("accidental").text
					if accidental == "sharp":
						accidental = 1
					else:
						accidental = -1
				height = pitch2height(note.find("pitch"), accidental)
				if height > highest:
					highest = height
				if lowest > height:
					lowest = height
				duration = note.find("duration").text
			if note.find("rest") is not None:
				height = 0
				duration = note.find("duration").text
			for i in range(int(int(duration) / dl)):
				if height != 0:
					mea_image[duration_point][height - 40] = 1
					if i != int(duration) - 1:
						rhythm_image[duration_point][height - 40] = 1
				duration_point += 1
		music_image.append([mea_image, rhythm_image])
	return music_image, highest, lowest
	
#music_image, highest, lowest = xml2list("test2.xml")
#print(highest, lowest)

global_highest = 0
global_lowest = 88
for f in glob.glob("after_xml_data/*.xml"):
	f_name = f.split("/")[-1].split(".")[0]
	try:
		music_image, highest, lowest = xml2list(f)
		if highest > global_highest:
			global_highest = highest
		if global_lowest > lowest:
			global_lowest = lowest
		np.save("music_numpy/%s.npy" % f_name, np.array(music_image))
	except:
		print(f)
print(global_highest, global_lowest)
	
 