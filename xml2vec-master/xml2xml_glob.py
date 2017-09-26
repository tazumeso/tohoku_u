# -*- coding: utf-8 -*-
"""Read MusicXML with multi part and write MusicXML with only melody part

xml2vecとvec2xmlのテスト用
複数パートのMusicXMLを読んでそこから一番上のパートとコード進行を抽出，
そのパートとコード進行だけのMusicXMLを作成して保存する
"""

import sys, os
import time
import re
import xml2vec as x2v
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
from bs4 import BeautifulSoup
import traceback


# mainで作ったMusicXMLにヘッダを追加して，改行，インデントを施す
def finalize(score):
    """Return finalized MusicXML
    
    Adds Header of MusicXML, Breaks lines and Indents
    """
    rough_string = ET.tostring(score, 'utf-8')
    rough_string = x2v.WriteHeader() + rough_string
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def change_xml(input_file, output_file):
    soup = BeautifulSoup(open(input_file, "r").read(), "lxml")
    piece_info, melody, chords = x2v.extract_music(soup)
    
    try:
        score = ET.Element("score-partwise")

        # 識別情報 (タイトル，作曲者など)
        x2v.WriteIdentification(score)

        # 各種デフォルト設定
        x2v.WriteDefaults(score)

        # パートリスト
        x2v.WritePartList(score)

        # 五線譜上の情報を書き込む
        x2v.WriteScore(score, piece_info, melody, chords)

        f = open(output_file, "w")
        f.write(finalize(score).encode('utf-8'))
        f.close()
        
    except:
        error_file = output_file.split(".")[0] + ".log"
        with open(error_file, 'a') as f:
            traceback.print_exc(file=f)
        

def  change_xml_dir(in_dir_name, out_dir_name):
    for input_file in glob.glob(in_dir_name + "/*.xml"):
        file_name = input_file.split("/")[-1]
        output_file = out_dir_name + "/" + file_name
        change_xml(input_file, output_file)
        
def mid2xml(input_file, output_file):
    mscore = "/Applications/MuseScore\ 2.app/Contents/MacOS/mscore"
    cmd = mscore + " -o %s %s" % (output_file, input_file)
    os.system(cmd)
    
def mid2xml_dir(input_dir, output_dir):
    for input_file in glob.glob(input_dir + "/*.mid"):
        file_name = input_file.split("/")[-1].split(".")[0]
        output_file =  "./%s/%s.xml" % (output_dir, file_name)
        imput_file = "./" + input_file
        mid2xml(input_file, output_file)
        

if __name__ == "__main__":

    argvs = sys.argv
    argc = len(argvs)
    if (argc != 4):
        print "Usage: python %s in_dir_name out_dir_name last_dir_name" % argvs[0]
        quit()
        
    in_dir_name = argvs[1]
    out_dir_name = argvs[2]
    last_dir_name = argvs[3]
    mid2xml_dir(in_dir_name, out_dir_name)
    change_xml_dir(out_dir_name, last_dir_name)

    print "Process Completed"
    
