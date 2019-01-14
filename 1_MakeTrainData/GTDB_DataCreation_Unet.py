#!/usr/bin/env python3
# coding: utf-8

___doc___ ="""{f}

GTDB_datacreation.py - Creat GTDataBase feature files

Usage:
	{f}	[-h|--help] [--outimg-dir=<outimg_dir>] [--window-height=<window_height>] [--scale=<scale>] [--stride=<stride>] [--area-csv=<area_type_CSV>] <GTDB_csv_file> <GTDB_imageDir> <OutputBlockDir>
""".format(f=__file__)

from docopt import docopt
import csv
import os
import numpy as np
from skimage import io, util, transform
from skimage.draw import polygon
from skimage.color import label2rgb
from skimage.morphology import dilation, disk
import random

# ---------------------
#  global variables
# ---------------------
GTDB_CSV_file = ""
GTDB_ImageDir = ""
Output_ImageDir = ""
Output_BlockDir = ""
Area_CSV_file = ""
Window_Height = 256
Window_Width  = 256
Window_Stride = 128
scale_rate = 0.25
bb_margin = 10 # bounding box margin
random_num = 7
AreaTypeDict = {}

def parse ():
	global GTDB_CSV_file
	global GTDB_ImageDir
	global Output_ImageDir
	global Output_BlockDir
	global Area_CSV_file
	global Window_Height 
	global Window_Width
	global scale_rate
	global Window_Stride 
	
	args=docopt(___doc___)

	GTDB_CSV_file = args['<GTDB_csv_file>']
	GTDB_ImageDir = args['<GTDB_imageDir>']
	Output_BlockDir = args['<OutputBlockDir>']
	if args['--outimg-dir']:
		Output_ImageDir = args['--outimg-dir']
	if args['--window-height']:
		Window_Height = int ( args['--window-height'] )
		Window_Width  = int ( args['--window-height'] )
		print ( 'window height = {}'.format(Window_Height) )
	if args['--scale']:
		scale_rate = float ( args['--scale'] )
		print ( 'scale = {}'.format(scale_rate) )
	if args['--stride']:
		Window_Stride = int ( args['--stride'] )
		print ( 'Window_Stride = {}'.format(Window_Stride) )
	if args['--area-csv']:
		Area_CSV_file = args['--area-csv']
		print ( 'AreaTypeCSVfile = {}'.format(Area_CSV_file) )

		
def Init_AreaType ( Area_CSV_file ):
	with open (Area_CSV_file, 'rt') as fin:
		cin = csv.reader(fin)
		flds = [row for row in cin]
	area_dict = dict ()
	for rcd in flds :
		area_dict [ rcd[1] ] = int ( rcd[0] )
	print( area_dict )
	return area_dict


def sheet ( sheet_file ):
	global scale_rate
	# 読み込み＆白黒反転
	page_image = np.bitwise_not( io.imread ( sheet_file, as_gray=True ) )
	# 縮小前に disk (radius = 1 / (2*scale)) で膨張処理 
	page_image_tmp = dilation ( page_image, selem=disk(int(1 / (scale_rate*2))) )
	page_image = transform.rescale ( page_image_tmp, (scale_rate, scale_rate) )
	horg, worg  = page_image_tmp.shape
	h, w  = page_image.shape
	print( "original image shape = {}, {}".format(worg,horg))
	print( "rescaled image shape = {}, {} : max = {}, min = {}".format(w,h,np.amax(page_image), np.amin(page_image)))
	ret_image = np.empty( (h, w), dtype=np.uint8 )
	ret_image = (page_image*255).astype(np.uint8)
	return ret_image


def rectangle(left, top, right, bottom, shape = None ):
	if left < 0: 
		left = 0
	if top < 0:
		top = 0
	if shape != None:
		if right >= shape[1]:
			right = shape[1]-1
		if bottom >= shape[0]:
			bottom = shape[0]-1
	rr, cc = [top, bottom, bottom, top], [left, left, right, right]
	return polygon( rr, cc )


def rectangle_line (left, top, right, bottom, line_height_default, shape = None ):
	y_center = (top+bottom)/2
	top = int ( y_center - line_height_default / 2 )
	bottom = int ( y_center + line_height_default / 2 )

	if left < 0: 
		left = 0
	if top < 0:
		top = 0
	if shape != None:
		if right >= shape[1]:
			right = shape[1]-1
		if bottom >= shape[0]:
			bottom = shape[0]-1
	rr, cc = [top, bottom, bottom, top], [left, left, right, right]
	return polygon( rr, cc )


def getbb ( blockList, linebbox=False ):

	left = blockList[0][0]
	top  = blockList[0][1]
	right = blockList[0][2]
	bottom = blockList[0][3]

	if linebbox != False:
		top = linebbox[1]
		bottom = linebbox[3]

	for i in blockList:
		left = min ( left, i[0] )
		top = min ( top, i[1] )
		right = max ( right, i[2] )
		bottom = max ( bottom, i[3] )
	
	rr, cc = rectangle ( left, top, right, bottom )
	return rr, cc

def output_images ( full_img, label_img, CharList, LineList, AreaList, output_filename ):

	global AreaTypeDict

	h, w = full_img.shape
	print( h, w )

	# ----- label only Math Character bounding box
	if any(AreaTypeDict) == False:
		
		for rcd in CharList :
			if int (rcd[7]) == 1 :
				block = list ( )
				block.append( [ int( max( (int(rcd[2])-bb_margin)*scale_rate, 0.0 )),
								int( max( (int(rcd[3])-bb_margin)*scale_rate, 0.0)),
								int( min( (int(rcd[4])+bb_margin)*scale_rate, w-1)),
								int( min( (int(rcd[5])+bb_margin)*scale_rate, h-1)) ] )
				rr, cc = getbb ( block, False )
				label_img[rr,cc] = 1.0
		
		label_img = label_img * ( full_img.astype(np.float16) / 255. )
		out_label = (label_img * 255).astype(np.uint8)
		label_img = dilation ( out_label, selem=disk(1) )
		
		if Output_ImageDir != "" :
			io.imsave ( Output_ImageDir + os.sep + output_filename + ".png", full_img )
			io.imsave ( Output_ImageDir + os.sep + output_filename + "_label.png", label_img )

	# ----- label all Area tags -------------
	else:
		for rcd in AreaList :
			block = list ( )
			block.append( [ int( max( (int(rcd[2])-bb_margin)*scale_rate, 0.0 )),
							int( max( (int(rcd[3])-bb_margin)*scale_rate, 0.0)),
							int( min( (int(rcd[4])+bb_margin)*scale_rate, w-1)),
							int( min( (int(rcd[5])+bb_margin)*scale_rate, h-1)) ] )
			rr, cc = getbb ( block, False )
			print ("*** {} *****".format(rcd))
			if rcd[0] == "Text":
				label_img[rr,cc,:] = 255

		# 	if len(rcd) <= 6:

		# 	else:
		# 		label_img[rr,cc,:] = AreaTypeDict[rcd[6]]

		# for rcd in CharList :
		# 	block = list ( )
		# 	block.append( [ int( (int(rcd[2])-bb_margin)*scale_rate),
		# 					int( (int(rcd[3])-bb_margin)*scale_rate),
		# 					int( (int(rcd[4])+bb_margin)*scale_rate),
		# 					int( (int(rcd[5])+bb_margin)*scale_rate) ] )
		# 	rr, cc = getbb ( block, False )
			
		# 	if int (rcd[7]) == 1 :
		# 		label_img[rr,cc,:] = AreaTypeDict ['MathChar']
		# 	elif rcd[12] != '0e81':
		# 		label_img[rr,cc,:] = AreaTypeDict ['TextChar']

		# output_label = label2rgb ( label_img[:,:,0], image=full_img, bg_label=0 )

		if Output_ImageDir != "" :
			io.imsave ( Output_ImageDir + os.sep + output_filename + ".png", full_img )
			io.imsave ( Output_ImageDir + os.sep + output_filename + "_label.png", label_img )
	# --------------------------------------


	# ----------------------
	#	上下左右にwindow size分のpadを入れる
	# ----------------------
	full_img = util.pad ( full_img, ( ( Window_Height, Window_Height ), ( Window_Height,Window_Height ) ), "wrap" );
	label_img = util.pad ( label_img, ( ( Window_Height, Window_Height ), ( Window_Height,Window_Height ) ), "wrap" );
	
	top = 0
	i = 0

	h = h + Window_Height * 2
	w = w + Window_Height * 2

	while top < h - Window_Height:
		bottom = top+Window_Height
		left = 0
		while left < w - Window_Width:

			for j in range ( random_num ):
				rand_offset_x = 0
				rand_offset_y = 0				
				if j != 0:
					rand_offset_x = random.randrange ( Window_Stride / 2 )
					rand_offset_x = min ( rand_offset_x, (w-Window_Width) - left );
					rand_offset_y = random.randrange ( Window_Stride / 2 )
					rand_offset_y = min ( rand_offset_y, (h-Window_Height) - top );

				right = left + Window_Width
				crop_label = util.crop ( label_img, ((top+rand_offset_y, h-bottom-rand_offset_y),(left+rand_offset_x, w-right-rand_offset_x)), False )
				crop_img   = util.crop ( full_img,  ((top+rand_offset_y, h-bottom-rand_offset_y),(left+rand_offset_x, w-right-rand_offset_x)), False )

				if crop_img.sum() != 0:
					filename = Output_BlockDir + os.sep + "images" + os.sep + output_filename + "_{0:05d}.png".format(i)
					io.imsave( filename, crop_img )
					filename = Output_BlockDir + os.sep + "groundtruth" + os.sep + output_filename + "_{0:05d}.png".format(i)
					io.imsave( filename, crop_label )
					i+=1

			left += Window_Stride
		top += Window_Stride

if __name__ == '__main__':

	parse()

x_data = np.empty((0,Window_Height*Window_Height), dtype='uint8')
y_data = np.empty(0, dtype='uint8')

# --- Check File and directories
if os.path.exists(GTDB_ImageDir) != True:
	print("ERROR: input image dir '{}' is not found".format(GTDB_ImageDir))
	exit()
if os.path.isfile(GTDB_CSV_file) != True:
	print("ERROR: input CSV file '{}' is not found".format(GTDB_CSV_file))
	exit()
if os.path.exists(Output_BlockDir) != True:
	print("ERROR: output block dir '{}' is not found".format(Output_BlockDir))
	exit()

# -- open AreaType CSV file (if needed)
if Area_CSV_file != "":
	if os.path.isfile (Area_CSV_file) != True:
		print( "Area CSV file '{}' is required but not found".format(Area_CSV_file))
		exit()
	AreaTypeDict = Init_AreaType ( Area_CSV_file )

# -- open CSV file
with open (GTDB_CSV_file, 'rt') as fin:
	cin = csv.reader(fin)
	flds = [row for row in cin]

n_sheet = 0
n_area = 0
n_line = 0
n_char = 0
fileline = 0
output_fileName = ""

prev_label = -1

for rcd in flds:
	if fileline == 0:
		fileline+=1
		continue

	if rcd[0] == 'Sheet' :
		if n_sheet != 0:
			output_images ( full_img, label, CharList, LineList, AreaList, output_fileName )
				
		image_fileName = "{}/{}".format(GTDB_ImageDir,rcd[2])
		output_fileName = rcd[2].replace(".png","")
		
		print ( "{}:Sheet : {}".format(fileline,image_fileName) )

		full_img = sheet ( image_fileName )

		h, w = full_img.shape
		label = np.zeros( (h,w), dtype=np.float16 )
		CharList = list()
		LineList = list()
		AreaList = list()

		n_sheet += 1
		n_line = 0
		
		print( x_data.shape, y_data.shape )

	if rcd[0] == 'Text' :
		'''
		rcs[1] : Area ID
		rcs[2],[3],[4],[5] : bounding box
		rcs[6] : Area Type (text)
		'''
		AreaList.append(rcd)

	if rcd[0] == 'Image' :
		'''
		rcs[1] : Area ID
		rcs[2],[3],[4],[5] : bounding box
		'''
		AreaList.append(rcd)

	if rcd[0] == 'Line' :
		'''
		rcs[1],[2],[3],[4] : left, top, right, bottom: bounding box
		'''
		LineList.append(rcd)

	if rcd[0] == 'Chardata' :
		'''
		rcd[1] id : char id in each line.
		rcd[2]-rcd[5] left, top, right, bottom: bounding box
		rcd[6] MathText: Math 1, ordinary text 0.
		rcd[7] Link label: 0:Horizontal, 1:RightSuperScript, 2:RightSubScript, 3:LeftSuperScript, 4:LeftSubScript, 5:Over, 6:Under.
		rcd[8] Parent Id: parent charcter id of the mathematical formula tree.
		rcd[9] ClusterId: used in Infty system (please ignore it).
		rcd[10] Cand Num: Always 1, this time.
		rcd[11] InftyCode: Unsigned Short code. Please see the InftyOcrCode.txt included in the package.
		'''
		CharList.append(rcd)
		
	fileline+=1

output_images ( full_img, label, CharList, LineList, AreaList, output_fileName )
