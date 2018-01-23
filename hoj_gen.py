#!/usr/bin/env/python
# coding=utf-8

# Python module import

import sys
import os
import math as ma
import argparse
import time as tM
import datetime as dT
import shutil 
import numpy as np

cur_version = sys.version_info[0]
print("\n")
print("Python Version: ", cur_version)
print("\n")
if( cur_version < 3 ):
	import cPickle as pickle
else:
	import _pickle as pickle

# Ppe module import
import load_skeleton as l_S
import data_extraction as d_e
import single_hoj_set as sh_set
import all_in_one_hoj as _hoj_transformation

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def main():

	comp_start_time = 0
	comp_end_time = 0

	_transformation_type = ''

	# Feedback data structure ( detailed information from the data computation to the main routine ).
	big_feedback = dict({\
		'_data_part_all':0,\
		'_processed_sets_all':0,\
		'_data_part_frames_all':0,\
		'_processed_frames_all':0,\
		'_number_of_divZero_error_all':0,\
		'_number_of_oCC_error_all':0})

	# Parse the command line options.
	clpDict = parseOpts( sys.argv )

	# Set the descriptor width depending on the chosen transformation.
	_number_of_bins_ = 0
	if( clpDict['_hoj'] is True ):
		_transformation_type = '_hoj'
		# Standard hoj descriptor width.
		_number_of_bins = 84 
		if( clpDict['_depth_measurement'] is not None ):
			# Double hoj descriptor width.
			_number_of_bins = _number_of_bins * 2
		if( clpDict['_body_parts'] is True ):
			# Standard hoj descriptor width + 4 bins for the body part correlation values.
			_number_of_bins = _number_of_bins + 4
	elif( clpDict['_ddd_points'] is True ):
		_transformation_type = '_ddd_points'
		# Set the desciptor width to the number of used joints multiplied by 3 ( the number of world coordinates )
		_number_of_bins = 3 * len(clpDict['_joint_index_list'])

	print( _number_of_bins )

	print("----------------------------------------------------------------------------------------------------------------------------")
	print("Init :: "+clpDict['_hoj_conf_string']+" was loaded." )
	print("----------------------------------------------------------------------------------------------------------------------------")

	# ----------------------------------------------------------------------------------------------------
	# Read configuration files 

	# Read list of missed or incomplete skelelton files.
	_missed_skeletons_fileHandler = open('conf/missed_skeletons','r+')
	_list_of_missed_skeletons = _missed_skeletons_fileHandler.readlines()
	[ _line.rstrip('\n') for _line in _list_of_missed_skeletons ]
	_missed_skeletons_fileHandler.close()

	# ----------------------------------------------------------------------------------------------------

	if( clpDict['_byte_object'] is False ):

		print("\n\n----------------------------------------------------------------------------------------------------------------------------")
		print("HojG_Main :: Start loading non binarized data.")
		print("----------------------------------------------------------------------------------------------------------------------------")

		# A index for the list 
		i = 0

		# List all skeleton files in the data directory.h3d
		_dir_list = os.listdir(clpDict['_path_name'])
		_dir_list = sorted(_dir_list)

		if( clpDict['_skeleton_name'] is not None):
			# Crop the list up to the point of the index of the given skeleton_name 
			# Then the list starts at this point
			_dir_list = _dir_list[_dir_list.index(clpDict['_skeleton_name']): ]

		# Start the clock.
		comp_start_time = tM.clock()

		# Devide the number of datasets by 4 to limit the amount of data in the memory. 
		_dir_list_parts = ma.floor(len(_dir_list) / 4)
		_low_dir_list_border = 0
		_upper_dir_list_border = _dir_list_parts

		for i in range(0,4):

			print("\n\nHojG_Main :: Start loading part #"+str(i+1)+" of 4.")

			print("LB: ",_low_dir_list_border)
			print("UB: ",_upper_dir_list_border)
			print("Round: ", i)

			if( i < 3 ):
				_parted_dir_list = _dir_list[int(_low_dir_list_border) : int(_upper_dir_list_border)]
			else:
				_parted_dir_list = _dir_list[int(_low_dir_list_border) : ]

			# Process the whole set one time. Read the data and store them in an internal structure.
			_internal_set_storage, _internal_set_name_storage, _number_of_previously_found_broken_sets = read_in_internal_data_storage(\
				clpDict['_path_name'],\
				_parted_dir_list,\
				_list_of_missed_skeletons,\
				clpDict['_action_list'],\
				clpDict['_ignore_tail'],\
				clpDict['_extraction_field'],\
				clpDict['_subsampling'],\
				clpDict['_sample_strategy'],\
				clpDict['_number_of_subframes'],\
				clpDict['_verbose'])
		 
			# Process the internal data structure and search for NaM's and zero values.
			_set_removal_list = preprocess_internal_data(\
				_internal_set_storage,\
				_internal_set_name_storage,\
				clpDict['_verbose'])

			# Remove newly found broken sets from the set list and return the cleaned internal storages.
			_internal_set_storage, _internal_set_name_storage = remove_broken_sets(\
				_internal_set_storage,\
				_internal_set_name_storage,\
				_set_removal_list,\
				clpDict['_verbose'] )

			# Write newly found broken sets to the configuration file.
			_number_of_newly_exluded_sets, _list_of_missed_skeletons = write_broken_sets_to_conf_file(\
				_set_removal_list,\
				_list_of_missed_skeletons,\
				'ms' )

			if( clpDict['_verbose'] ):
				_exluded_sets = _number_of_newly_exluded_sets + _number_of_previously_found_broken_sets
				print("HojG_Main :: Number of exluded sets: %5d" %_exluded_sets)

			# Store temporary pickles. Each pickle contains a part of the dataset.
			store_pickles( _internal_set_storage, "data_part", 'tmp', i )
			store_pickles( _internal_set_name_storage, "name_part", 'tmp', i )

			# Set the new part borders.
			_low_dir_list_border = _upper_dir_list_border + 1
			_upper_dir_list_border = _upper_dir_list_border + _dir_list_parts

			# Clean the data storage
			_internal_set_storage = None
			_internal_set_name_storage = None

			print("\n\nHojG_Main :: Finish loading part #"+str(i+1)+" of 4.")

		print("\n\n----------------------------------------------------------------------------------------------------------------------------")
		print("HojG_Main :: Step 1 - Loading data is now finished.")
		print("----------------------------------------------------------------------------------------------------------------------------")

		print("\n\nHojG_Main :: Start loading previously binarized data objects.")

	_data_pickel = []

	# Update the feedback structure for the whole datset.
	big_feedback['_data_part_all'] = 0

	for i in range(0,4):

		print("\n\nHojG_Main :: Load binarized data object "+str(i+1)+" of 4.")	
		print("----------------------------------------------------------------------------------------------------------------------------")

		if( clpDict['_byte_object'] is True and clpDict['_byte_object_path'] is not 'tmp_data/'):
			_part_name = clpDict['_byte_object_path']+"data_part_"+str(i)+".bin_data_tmp"
			_data_part = load_pickles( _part_name, "data" )
			_part_name = clpDict['_byte_object_path']+"name_part_"+str(i)+".bin_data_tmp"
			_name_part = load_pickles( _part_name, "name" )
		else:
			_part_name = "tmp_data/data_part_"+str(i)+".bin_data_tmp"
			_data_part = load_pickles( _part_name, "data" )
			_part_name = "tmp_data/name_part_"+str(i)+".bin_data_tmp"
			_name_part = load_pickles( _part_name, "name" )

		# Compute the hoj descriptor from the data part.
		_data_container, _occlusion_name_list, _nan_name_list, feedback = compute_hoj_using_the_data_parts(\
			_data_part,\
			_name_part,\
			_hoj_transformation,\
			_transformation_type,\
			clpDict['_joint_index_list'],\
			_number_of_bins,\
			clpDict['_error_handling_strategy'],\
			#TODO .... die klassen als command line argument.
			61,\
			clpDict['_verbose'],\
			clpDict['_depth_measurement'],\
			clpDict['_body_parts'],\
			clpDict['_loCoords'])

		# Give free all the memory for the h. 
		data_part = None
		name_part = None

		# Give feedback of the computation from this data part. 
		_data_part_sets = feedback['_data_part_sets']
		_processed_sets = feedback['_processed_sets']
		_data_part_frames = feedback['_data_part_frames']
		_processed_frames = feedback['_processed_frames']
		_number_of_divZero_error = feedback['_number_of_divZero_error']
		_number_of_oCC_error = feedback['_number_of_oCC_error']
		if( clpDict['_error_handling_strategy'] is 'skip_set' ):
			print('\n\r')
			print("CDP :: %d sets of %d sets in total have been transformed." %(_processed_sets, _data_part_sets ) )
			print("CDP :: %d sets skipped in this part." %( _data_part_sets-_processed_sets ) )
		elif( clpDict['_error_handling_strategy'] is 'skip_frame' ):
			print('\n\r')
			print("CDP :: %d sets of %d sets in total have been transformed." %(_processed_sets, _data_part_sets ) )
			print("CDP :: %d frames of %d frames in total have been transformed." %(_processed_frames, _data_part_frames ) )
			print("CDP :: %d frames skipped in this part." %( _data_part_frames-_processed_frames ) )

		# Concatenate the data parts together in one list.
		_data_pickel.extend(_data_container[:])

		# Update the feedback structure for the whole data set.
		big_feedback['_data_part_all'] = big_feedback['_data_part_all'] + _data_part_sets
		big_feedback['_processed_sets_all'] = big_feedback['_processed_sets_all'] + _processed_sets
		big_feedback['_data_part_frames_all'] = big_feedback['_data_part_frames_all'] + _data_part_frames
		big_feedback['_processed_frames_all'] = big_feedback['_processed_frames_all'] + _processed_frames
		big_feedback['_number_of_divZero_error_all'] = big_feedback['_number_of_divZero_error_all'] + _number_of_divZero_error
		big_feedback['_number_of_oCC_error_all'] = big_feedback['_number_of_oCC_error_all'] + _number_of_oCC_error

	store_pickles(_data_pickel, "data", 'final', 0)

	comp_end_time = tM.clock()
	timeDiff = (dT.timedelta(seconds=comp_end_time - comp_start_time)).total_seconds()

	print("\n")
	print("----------------------------------------------------------------------------------------------------------------------------")
	print("HojG_Main :: Step 2 - Processing data is now finished.")
	print("----------------------------------------------------------------------------------------------------------------------------")

	# Give feedback to the whole dataset.
	print("\n\nSome statistics: -----------------------------------------------------------------------------------------------------------")
	print("\n")
	print("Number of sequences in dataset             : ", big_feedback['_data_part_all'])
	print("Number of computed sequences               : ", big_feedback['_processed_sets_all'])
	print("Number of excluded sequences               : ", ( big_feedback['_data_part_all'] - big_feedback['_processed_sets_all'] ) )
	print("Number of frames in dataset                : ", big_feedback['_data_part_frames_all'])
	print("Number of computed frames                  : ", big_feedback['_processed_frames_all'])
	print("Number of excluded frames                  : ", ( big_feedback['_data_part_frames_all'] - big_feedback['_processed_frames_all'] ) )
	print("Number of datasets with NaN entrys         : ", big_feedback['_number_of_divZero_error_all'])
	print("Number of datasets wth occluded body parts : ", big_feedback['_number_of_oCC_error_all'])
	print("----------------------------------------------------------")
	print("Used time for the computation              : ", timeDiff )

	_missed_skeletons_fileHandler.close()
	# index_exceptions_fH.close()
	# inconsistency_exception_fH.close()
	# nan_exception_fH.close()
	print("\n\n")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
 
def compute_hoj_using_the_data_parts( 
	_data_part, 
	_name_part, 
	_hoj_transformation,
	_transformation_type,
	_joint_index_list,  
	_number_of_bins_, 
	_error_handling_strategy='skip_set', 
	_classes=61, 
	_verbose=False, 
	_depth_measurement=None, 
	_body_parts=False,
	_loCoords=False ):

	# Just for the fancyness of the commandline output.
	print("\n")

	# The final data container.
	_data = []

	# The index for requesting names from the _name_part.
	# The size of the name part is the same as the size of the data part.
	# This means the name resolution is on set level, not frame level.
	_name_index = 0 

	# Get the number of sets in the data part.
	_data_part_sets = len(_data_part[0])
	_processed_sets = 0

	# Get the number of frames in this data part. ( It's the number of frames after subsampling. )
	_data_part_frames = 0
	for _obj in _data_part[0]:
		_data_part_frames = _data_part_frames + len(_obj)
	_processed_frames = 0
	
	# Create a name list for sets with nan errors
	_nan_name_list = []
	_number_of_divZero_error = 0

	# Create a name list for sets with occlusion errors
	_occlusion_name_list = []
	_number_of_oCC_error = 0

	# Error flags for the hoj computation.
	_divZero_error_occured = False
	_oCC_error_occured = False

	# Feedback data structure ( detailed information from the data computation to the main routine ).
	feedback = dict({\
		'_data_part_sets':_data_part_sets,\
		'_processed_sets':0,\
		'_data_part_frames':_data_part_frames,\
		'_processed_frames':0,\
		'_number_of_divZero_error':0,\
		'_number_of_oCC_error':0,})

	# Step trough the sets of this data part.
	for _set in _data_part[0]:
	
		# Create a hoj_set with the correct size ( number_of_set_files x descriptor length )
		_set_of_hojes = np.zeros( ( len(_set), _number_of_bins_ ) )
		_set_of_hojes_index = 0

		# All labels of the set
		hoj_labels = []

		# Get the name of the actual set.
		_name = _name_part[0][_name_index]

		# Step through the frames of this part.
		for _frame in _set:

			# Generate a name descriptor for this frame.
			_labels= np.zeros(int(_classes))

			# Get the joints from the frame.
			list_of_joints = _frame.get_ListOfJoints()

			# If this try catches an IndexError it means a frame has missing joints. We will now decide how we will process.
			# A) Will we delete only the frame or B) will we delete the whole set.
			# Both strategies have something for their own.
			# If we delete only the frame(s), because their could be more than one single frame in this set also whole sequences of missing parts,
			# we get noise data. Maybe real strange sequences of people instantly switching rotation and/or position.
			# Otherwise we will lose a whole set and maybe their is a benefit in the noise data.
			# We should give em a try and implement booth strategies.
			# Lets go.
			try:
				# A flag if the hoj comutation fails.
				hoj_true = False

				if( _transformation_type is '_hoj' ):
					# compute_hoj3d( full joint list, hip center, spine, hip right, hip left .... )
					hoj3d_frame,time = _hoj_transformation.compute_hoj3d(\
						list_of_joints,\
						list_of_joints[0],\
						list_of_joints[1],\
						list_of_joints[16],\
						list_of_joints[12],\
						joint_indexes=_joint_index_list,\
						use_triangle_function=True,\
						n_time=0.0,\
						_depth_measurement=_depth_measurement,\
						_body_parts=_body_parts)
				elif( _transformation_type is '_ddd_points' ):
					hoj3d_frame,time = _hoj_transformation.get_joint_coordinates(\
						list_of_joints,\
						list_of_joints[0],\
						list_of_joints[1],\
						list_of_joints[16],\
						list_of_joints[12],\
						joint_indexes=_joint_index_list,\
						n_time=0.0,\
						local=_loCoords)		
			
				# If the computation of the hoj returns a -1 for time ( Error :: Division by Zero ) then use strategy to skip frame or set.
				if( time is -1 ):
					# Increase nan counter.
					_number_of_divZero_error = _number_of_divZero_error + 1
					# Append name to list
					_nan_name_list.append(_name)

					print("CDP :: !!! Actual set " + _name + " has at least one NaN value error in the hoj transformation!!!\n")
					# Set flag.
					_divZero_error_occured = True

			# If the computation of the hoj returns an IndexError ( Error :: Missing skeleton joints ) then use strategy to skip frame or set.
			except IndexError:

					# Increase occlusion counter.
					_number_of_oCC_error = _number_of_oCC_error + 1
					# Append name to list
					_occlusion_name_list.append(_name)

					print("CDP :: !!! Actual set " + _name +" has joint index error(s) in the hoj transformation!!!")
					# Set flag.
					_oCC_error_occured = True

			# If an error occured?
			if( _divZero_error_occured or _oCC_error_occured ):

				# Which strategie? Skip the frame or skip the whole set?
				if( _error_handling_strategy is 'skip_set' ):
					if( _verbose is True ):
						print("CDP :: Use strategy: skip_set.\n\n")
					break
				elif( _error_handling_strategy is 'skip_frame'): 
					if( _verbose is True ):
						print("CDP :: Use strategy: skip_frame.\n\n")				
					_divZero_error_occured = False
					_oCC_error_occured = False
			# No error at all.
			else:
				# hoj3d_frame = np.reshape( hoj3d_frame,( 1, len( hoj3d_frame ) ) )

				# Add the hoj frame to the set storage.
				_set_of_hojes[_set_of_hojes_index] = hoj3d_frame[:]
				# Increment the set index.
				_set_of_hojes_index = _set_of_hojes_index + 1

				# Extract the class label for the frame
				idx = int(_name[-3:])
				# Store the class of that hoj
				_labels[idx] = 1
				# Add the label for this frame to the storage for the set labels.
				hoj_labels.append(_labels)

				# Print the set computation progress
				sys.stdout.write("\rCDP :: %d frames of %d frames in total have been transformed." %(_processed_frames+1, _data_part_frames ) )
				sys.stdout.flush()					
				_processed_frames = _processed_frames + 1

		# We will not store the actual set if an error occured and 'skip_set' is chosen as strategy.
		if( _divZero_error_occured is False and _oCC_error_occured is False ):
			# Instantiate a single_hoj_set container. 
			h_set = sh_set.single_hoj_set()
			# Store the hoj_set to this container.
			h_set.set_hoj_set( _set_of_hojes )
			# Store the labels in the container.
			h_set.set_hoj_label( hoj_labels )
			# Give the container the name of this set.
			h_set.set_hoj_set_name( _name ) 
			# Append the last set container to the _data container for this data part.
			_data.append( h_set )
			# Increment the processed sets
			_processed_sets = _processed_sets + 1
		else:
			# The set will not be stored.
			# We only reset the flags.
			_divZero_error_occured = False
			_oCC_error_occured = False

		# Increase the name index for the name extraction out of the name data object.
		_name_index = _name_index + 1

	feedback['_processed_sets'] = _processed_sets
	feedback['_processed_frames'] = _processed_frames
	feedback['_number_of_divZero_error'] = _number_of_divZero_error
	feedback['_number_of_oCC_error'] = _number_of_oCC_error

	return _data, _occlusion_name_list, _nan_name_list, feedback

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
 
def write_broken_sets_to_conf_file( _set_removal_list, _list_of_missed_skeletons, _type='ms' ):
	# Copy the old list
	_new_missed_skeleton_list = _list_of_missed_skeletons[:]

	# Extend the old list by the newly found broken sets.
	for _set in _set_removal_list:
		if _set not in _list_of_missed_skeletons:
			_new_missed_skeleton_list.append(_set)


	# If we found new broken skeletons which are not part of the stored missed skeleton list delete the old list and write a new one.
	_number_of_newly_exluded_sets = len( _new_missed_skeleton_list ) - len(_list_of_missed_skeletons)
	if ( _number_of_newly_exluded_sets > 0 ):

		# Open the file
		if( type is 'ms'):
			_missed_skeletons_fileHandler = open('conf/missed_skeletons','r+')
		elif( type is 'nan'):
			_missed_skeletons_fileHandler = open('conf/nan_list','r+')
		elif( type is 'occ'):
			_missed_skeletons_fileHandler = open('conf/occ_list','r+')
		
		# Remove the old content from the file.
		_missed_skeletons_fileHandler.truncate(0)

		_new_missed_skeleton_list.sort()

		# Write new content to the list.
		for _missed_skeleton in _new_missed_skeleton_list:
			_missed_skeletons_fileHandler.write(_missed_skeleton)

		# Close the file
		_missed_skeletons_fileHandler.close()

	return _number_of_newly_exluded_sets, _new_missed_skeleton_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def remove_broken_sets(  _internal_set_storage, _internal_set_name_storage, _set_removal_list, _verbose=False ):

	_set_index = 0
	_new_internal_list = []
	_new_name_list = []

	# Remove all sets from the internal structure which are marked as broken.
	for _set in _internal_set_name_storage:
		
		if _set not in _set_removal_list:
			# If the set wasn't marked for removel append it to the unbroken set list.
			_new_internal_list.append(_internal_set_storage[_set_index])
			# The same for the name.
			_new_name_list.append(_set)

		_set_index = _set_index + 1	

	# This storages should only contain the unbroken set data now.
	_internal_set_storage = _new_internal_list[:]
	_internal_set_name_storage = _new_name_list[:]

	if( _verbose is True ):
		print("RBS :: Remaining Sets  : ", _internal_set_name_storage)

	# Return the cropped data
	return _internal_set_storage, _internal_set_name_storage

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Evaluate the data integrity 
def preprocess_internal_data( _internal_set_storage, _internal_set_name_storage, _verbose=False ):
	
	_name_index = 0
	_set_removal_list = []
	_break_set = False

	# Traverse through the set data structure. Set by set.
	for _set in _internal_set_storage:
		_break_set = False

		# Step through the frames of the set. Frame by frame.
		for _frame in _set:

			# Get the list of joints from the actual frame
			_joints_of_the_frame = _frame.get_ListOfJoints()

			# ------------ Test for missed skeleton data ------------ #

			# Step through the joints of this frame. Joint by joint.
			for _joint in _joints_of_the_frame:

				# Get the world coordinates of the joint
				x_world,y_world,z_world = _joint.get_WorldJoint()
				
				# If one of the world coordinates is Zero or NaN
				if( ( x_world == 0 or ma.isnan(x_world) is True ) or ( y_world == 0 or ma.isnan(y_world) is True ) or ( z_world == 0 or ma.isnan(z_world) is True ) ):

					# Add the set name to the removel list. ( For the subsequent removel of all broken sets. )
					_new_set_to_remove = _internal_set_name_storage[_name_index]+'\n'
					_set_removal_list.append( _new_set_to_remove )
					_break_set = True

					# If a broken frame was detected leave the joint loop ....
					break
	
			if( _break_set is True ):
				# ..... and leave the frame loop.
				break

		# Increment the name index 
		_name_index = _name_index + 1

	if( _verbose is True ):
		print("PID :: List of excluded sets: ", _set_removal_list )

	return _set_removal_list

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def read_in_internal_data_storage( _path_name, _dir_list, _list_of_missed_skeletons_, _action_list, _ignore_tail, _extraction_field, _subsampling=False, _sample_strategy='random', _number_of_subframes=8, _verbose=False ):

	i = 0
	_name = ""
	_internal_set_storage = []
	_internal_set_name_storage = []
	_number_of_previously_found_broken_sets = 0

	# Instatiate the data extraction class.
	de_Class = d_e.data_extraction()

	# Step through the listed files in the dir_list
	for file in _dir_list[i:]:

		if( '.skeleton' in file ):
			# Check if the choosen file is in the list of files with missed or incomplete data.
			if( check_for_file_consistence( file.split(".")[0], _list_of_missed_skeletons_ ) is False ):

				# Build the whole filename with absolute path
				_skeleton_filename = _path_name + file

				# Check for an action we want ( stored in the action list )
				if( is_action( _action_list, _skeleton_filename ) ):

					_all_skeleton_frames = []

					# ----------------------------------------------------------------------------------------------------
					# Open the skeleton file if it exist.
					if( os.path.isfile( _skeleton_filename ) == True ):
						print("RID :: Skeleton file: ", _skeleton_filename )
						_skeleton_fileHandler = open( _skeleton_filename , 'r')

						# Read the data from the skeleton file for the whole sequence
						_all_skeleton_frames = l_S.read_skeleton_data( _skeleton_fileHandler, _ignore_tail, _verbose )[:]

						if( _subsampling is True ):
							# Because of the memory limitation subsampling will be done here.
							_all_skeleton_frames = de_Class.frame_subsampling( _all_skeleton_frames, _sample_strategy, _number_of_subframes, _verbose )

						# if( _data_extraction is True )
						# Because of the limitation of memory it's recommended to extract onle the data fields you need.
						# Options: world, depth values, color values, depth and color, depth and world, color and world 
						#_all_skeleton_frames = de_Class.extract_data( _all_skeleton_frames, _extraction_field )

					else:
						print("\nRID :: No skeleton file with name: ", _skeleton_filename )
						print("RID :: Leave script now.\n")
						exit(0)

					if os.path.exists(os.path.splitext(file)[0]):
						os.makedirs(os.path.splitext(file)[0])

					# Store the unbroken set in the internal storage.
					_internal_set_storage.append( _all_skeleton_frames )
					_internal_set_name_storage.append(file.split('.')[0])

			else:
				# If the set was broken increment the number of broken sets counter.
				_number_of_previously_found_broken_sets = _number_of_previously_found_broken_sets + 1

	index = 0
	if( _verbose is True ):
		for index in range(index, len(_internal_set_storage) ):
			print("RID :: Set name                    : ", _internal_set_name_storage[index] )
			print("RID :: Number of frames in this set: ", len(_internal_set_storage[index]) )

	return _internal_set_storage, _internal_set_name_storage, _number_of_previously_found_broken_sets

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# A small function to skip actions which are not in the action list
def is_action( _action_list_, _skeleton_filename_ ):
	# If an action_list is given 
	if( _action_list_ is not None ):
		for key in _action_list_:
			if( key in _skeleton_filename_ ):
				# If the action of the skeleton file is in the action list.
				return True
	# If no action list is given
	else:
		return True

	# If the action of the skeleton file is not in the action list.
	return False		

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# A small function to check wether a file with incomplete data was choosen or not.
def check_for_file_consistence( _skeleton_filename_, _list_of_missed_skeletons_ ):
	# Crop the choosen filepath to only the name
	#_skeleton_filename_ = _skeleton_filename_.split('/')[1].split('.')[0]

	# Step trough the list of known files with missed data.
	for name in _list_of_missed_skeletons_:
		# Remove the trialing backslash from this name in the list.
		name = name.rstrip('\n')

		#if the name is in the list return true
		if _skeleton_filename_ == name:
			return True
	# otherwise return false		
	return False

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Store pickling objects

def store_pickles( _data, _name, _type='tmp', _part='0' ):

	if( _type is 'tmp' ):
		path='tmp_data/'
		f_name = _name + '_' + str(_part)
	elif( _type is 'final' ):
		path='pickles/'
		f_name = _name

	if( os.path.exists(path) is False ):
		os.makedirs(path)

	if( _type is 'tmp' ):
		store_data_path=str(path)+str(f_name)+".bin_data"+"_tmp"
	elif( _type is 'final' ):
		store_data_path=str(path)+str(f_name)+".bin_data"

	with open(store_data_path, 'wb') as f:
		pickle.dump(_data, f, protocol=2)
		print("SP :: Serialized "+ _name + " object was stored.")	

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Load previously stored pickle data objects 

def load_pickles( _path, _part ):

	_data = None
	_labels = None

	with open( _path, 'rb') as f:
		_data = pickle.load(f)
#			_data = pickle.load(f, encoding='latin1')
		_dir_counter = len(_data)
		number_of_files = 0
		for _obj in _data:
			#number_of_files = number_of_files + len(_obj.get_hoj_set())
			number_of_files = number_of_files + 1

# TODO: Die Ausgabe stimmt nicht. 
		if( _part is 'data'):
			print("LP :: "+str(number_of_files)+" sets from "+str(_dir_counter)+" sets in total were loaded from "+str(_path))
		elif( _part is 'name'):
			print("LP :: "+str(number_of_files)+" names from "+str(_dir_counter)+" names in total were loaded from "+str(_path))
		else:
			print("LP :: "+str(number_of_files)+" files from "+str(_dir_counter)+" directories were loaded from "+str(_path))

	return _data, _dir_counter

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# Parse the command line arguments
def parseOpts( argv ):


	_path_name = ""
	_byte_object = False
	_byte_object_path = ""
	_dataset_name = ""
	_skeleton_name = None
	_action_list = []
	_ignore_tail = False

	_joint_index_list = None
	_extraction_field = 'world'
	_subsampling = False
	_number_of_subframes = 8	
	_sample_strategy = "first"
	_error_handling_strategy = "skip_set"
	_hoj = True
	_ddd_points = False
	_loCoords = False  
	_depth_measurement = None
	_body_parts = False
	_hoj_conf_string = ""

	_remove_tmp_data_objects = False

	_verbose = False

	clpDict = {\
		'_path_name':_path_name,\
		'_byte_object':_byte_object,\
		'_byte_object_path':_byte_object_path,\
		'_skeleton_name':_skeleton_name,\
		'_action_list':_action_list,\
		'_ignore_tail':_ignore_tail,\
		'_joint_index_list':_joint_index_list,\
		'_extraction_field':_extraction_field,\
		'_subsampling':_subsampling,\
		'_number_of_subframes':_number_of_subframes,\
		'_sample_strategy':_sample_strategy,\
		'_error_handling_strategy':_error_handling_strategy,\
		'_remove_tmp_data_objects':_remove_tmp_data_objects,\
		'_hjo':_hoj,\
		'_ddd_points':_ddd_points,\
		'_loCoords':_loCoords,\
		'_depth_measurement':_depth_measurement,\
		'_body_parts':_body_parts,\
		'_hoj_conf_string':_hoj_conf_string,\
		'_verbose':_verbose}

	# generate parser object
	parser = argparse.ArgumentParser()

	# Dataset Control 
	parser.add_argument('-path', '--path_name', action='store', dest='_path_name', help='The path to the dataset.')
	parser.add_argument('-bo','--byte_object', action='store_true', dest='_byte_object', help='If you want to read data from a previously computed byte object.')
	parser.add_argument('-bop','--byte_object_path', action='store', dest='_byte_object_path', help='The path to the byte_object.')

	# Dataset Shape Control 
	parser.add_argument('-pn', '--part_name', action='store', dest='_skeleton_name', help='The name of the dataset in the path.')
	parser.add_argument('-aL', '--action_list', action='store', dest='_action_list', help='A list of actions in the form: -aL A001,A002,A003,...  ')
	parser.add_argument('-iT', '--ignore_tail', action='store_true', dest='_ignore_tail', default='False', help='The last frame in each set will be skipped if this flag is enabled.' )
	parser.add_argument('-jil', '--joint_index_list', action='store', dest='_joint_index_list', help='A list of joints which will be used for the computation.')
	parser.add_argument('-eF', '--extraction_fields', action='store', dest='_extraction_field', help='Choose the data field you want to extract [ world ( default ) | depth | color | deco | woco | all ] ')
	parser.add_argument('-sub', '--subsampling', action='store_true', dest='_subsampling', help='Turn on if you want to subsample the original data. ( Use -sf to define the number of subsamples per set and -ss to define the subsampling strategy. Defaults are 8 and first. ) ')
	parser.add_argument('-sf', '--subframes', action='store', dest='_number_of_subframes', help='The number of frames per set in the training.')
	parser.add_argument('-ss', '--sample_strategy', action='store', dest='_sample_strategy', help='Defines the strategy of the set subsampling. [ first ( default ) | mid | last | random ]')
	parser.add_argument('-cs', '--computation_strategy', action='store', dest='_error_handling_strategy', help='Defines the strategy of the error handling in the hoj computation [ skip_set | skip_frame ]. See the Readme for further information.')

	# Output Control
	parser.add_argument('-3d', '--3d_points', action='store_true', dest='_ddd_points', help='If this is set to true, the output will be in the form of 3d points in the local skeleton coordinate system.')
	parser.add_argument('-hoj', '--hoj', action='store_true', dest='_hoj', help='If this is set to true, the output will be in the form of a histogram of oriented joints ( using the sphere and parts extention ).')
	# TODO: Readme eintrag für das depth_measurment und die body parts schreiben.
	parser.add_argument('-dM', '--depth_measurement', action='store', dest='_depth_measurement', help='Do you want to use the original hoj ( None ) or a hoj with a second layer ( cylinder | sphere )? [ None (default) | cylinder | sphere ]. See the Readme for further information.')
	parser.add_argument('-bP', '--body_parts', action='store_true', dest='_body_parts', help='Do you want to compute body part correlations? [ False ( default ) | True ]. See the Readme for further information.')
	parser.add_argument('-loCoord', '--local_coord', action='store_true', dest='_loCoords', help='If this is set to true, the output will be in the form of 3d points in the global camera coordinate system.')
	parser.add_argument('-rTDO', '--remove_temp_data_objects', action='store_true', dest='_remove_tmp_data_objects', help='Do you want to remove the tmp_data objects in the tmp_data/ folder after the hoj computation?')

	# Global control parameters 
	parser.add_argument('-v', '--verbose', action='store_true', dest='_verbose', default='False', help='True if you want to listen to the chit-chat.')

	# Finally parse the command line.
	args = parser.parse_args()

	print("\n\nInformation: ---------------------------------------------------------------------------------------------------------------")

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Dataset Control 
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# The path to the data.
	if( args._path_name ):
		clpDict['_path_name'] = args._path_name
	else:		
		clpDict['_path_name'] = "skeleton/"

	# Use a previously computed data object ( data + labels )
	if( args._byte_object is True ):
		clpDict['_byte_object'] = True
		if( args._byte_object_path):
			clpDict['_byte_object_path'] = args._byte_object_path
		else:
			clpDict['_byte_object_path'] = "tmp_data/"

	# If a dataset name is given ( the list of datasets to compute then starts at this position )
	if( args._skeleton_name ):
		clpDict['_skeleton_name'] = str(args._skeleton_name + ".skeleton")

	# A list to build a subset from the data.
	if( args._action_list ):
		clpDict['_action_list'] = args._action_list.split(",")
	else:
		clpDict['_action_list'] = None

	if( args._ignore_tail ):
		clpDict['_ignore_tail'] = True
	else:
		clpDict['_ignore_tail'] = False

	# A list which specify which joints will be used in the subsequent computation.
	if( args._joint_index_list is not None ):
		joint_index_list_ = []
		joint_index_ = args.joint_index_list.split(",")
		for obj_ in joint_index_:
			joint_index_list_.append(int(obj_))
		clpDict['_joint_index_list'] = joint_index_list_
	else:
		# Used joints in the paper: 3, 5, 9, 6, 10, 13, 17, 14, 18, 12, 16
		# head 0 | l elbow	1 | r elbow	2 | l hand 	3 | r hand 	4 | l knee 	5 | r knee 	6 | l feet 	7 | r feet 	8
		# Aktuelle Joint List ohne hips
		clpDict['_joint_index_list'] = [3, 5, 9, 6, 10, 13, 17, 14, 18]

	# Because of the main memory limitation it is recommended to only extract the joint data you need. ( world coordinates, depth values, color values, or color)
	if( args._extraction_field ):
		clpDict['_extraction_field'] = args._extraction_field
	else:
		clpDict['_extraction_field'] = "world"

	if( args._subsampling ):
		clpDict['_subsampling'] = True

		# The number of subframes u will use from each subset.
		if args._number_of_subframes:
			clpDict['_number_of_subframes'] = int(args._number_of_subframes)
		else:
			clpDict['_number_of_subframes'] = 8
		
	# Which strategy will u use for the set subsampling?
	if args._sample_strategy: 
		clpDict['_sample_strategy'] = args._sample_strategy
	else:
		clpDict['_sample_strategy'] = "first"

	# Which strategy will u use for the set subsampling?
	if args._error_handling_strategy: 
		clpDict['_error_handling_strategy'] = args._error_handling_strategy
	else:
		clpDict['_error_handling_strategy'] = "skip_set"

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Output Control
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	# Will you store the data object after the first time u create it?
	if args._remove_tmp_data_objects:
		clpDict['_remove_tmp_data_objects'] = True
	else:
		clpDict['_remove_tmp_data_objects'] = False

	# Will u use hoj format as output or 3d points?
	if( args._hoj is True ):
		clpDict['_hoj'] = True

		if( args._depth_measurement is not None ):
			clpDict['_depth_measurement'] = str(args._depth_measurement)

		if( args._body_parts is True ):
			clpDict['_body_parts'] = True

		clpDict['_ddd_points'] = False
	elif( args._ddd_points is True ):
		clpDict['_hoj'] = False
		clpDict['_ddd_points'] = True
		if( args._loCoords ):
			clpDict['_loCoords'] = True
	else:
		clpDict['_hoj'] = True
		clpDict['_depth_measurment'] = None
		clpDict['_body_parts'] = False
		clpDict['_ddd_points'] = False
		clpDict['_loCoords'] = False

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Global control parameters
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	if( args._verbose ):
		clpDict['_verbose'] = args._verbose
	else:
		clpDict['_verbose'] = False

	# ----------------------------------------------------------------------------------------------------------------------------------------------------------
	# Print the configuration
	# ----------------------------------------------------------------------------------------------------------------------------------------------------------

	print ("\nConfiguration:")
	print ("----------------------------------------------------------------------------------------------------------------------------")
	if clpDict['_byte_object'] is False:
		print ("Data location             : ", clpDict['_path_name']  )
	else:
		print ("Pickle object             : ", clpDict['_byte_object']  )
		if clpDict['_byte_object_path'] is not None:
			print ("Pickle object path        : ", clpDict['_byte_object_path']  )
		else:
			print ("Pickle object path        : tmp_data/"  )
	if( clpDict['_skeleton_name'] is not None ):
		print ("Start with Set            : ", clpDict['_skeleton_name'] )
	if( clpDict['_action_list'] is not None ):
		print ("Action list            : ", clpDict['_action_list'] )
	if( clpDict['_subsampling'] is True):
		print ("('Subsampling               : ', Activated ")
		print ("Subsampling strategy      : ", str(clpDict['_sample_strategy']))
		print ("Number fo Subframes       : ", int(clpDict['_number_of_subframes']))
	print("Error handling strategy   : ", clpDict['_error_handling_strategy'])
	print("Used joints               : ", clpDict['_joint_index_list'] ) 
	if( clpDict['_ignore_tail'] is True ):
		print ("Ignore tail               : ", clpDict['_ignore_tail'] )
	if( clpDict['_loCoords'] is True ):
		_oO ="Camera Coordinate System"
		print("Coordinate Origin             : ", _oO )
	else:
		_oO ="Skeleton Coordinate System"
		print("Coordinate Origin         : ", _oO )

	if( clpDict['_hoj'] is True and clpDict['_ddd_points'] is False  ):
		_cOPt = "Histogram of Oriented Joints"	
		
		if( clpDict['_depth_measurement'] is not None ):
			if( clpDict['_depth_measurement'] is 'sphere'):
				_cOPt = _cOPt + " with sphere extension"

			if( clpDict['_depth_measurement'] is 'cylinder'):
				_cOPt = _cOPt + " with cylinder extension"

		if( clpDict['_body_parts'] is True ):
			if( clpDict['_depth_measurement'] is not None ):
				_cOPt = _cOPt + " and body part correlation extension"
			else:
				_cOPt = _cOPt + " with body part correlation extension"

		clpDict['_hoj_conf_string'] = _cOPt
		print ("Computation Output        : ", _cOPt)
	if( clpDict['_hoj'] is False and clpDict['_ddd_points'] is True  ):
		_cOPt = "3D Points" 
		print ("Computation Output        : ", _cOPt)
		clpDict['_hoj_conf_string'] = _cOPt

	if clpDict['_remove_tmp_data_objects']:
		print ("Remove temp data objects? : ", clpDict['_remove_tmp_data_objects'] )
	print ("verbose                   : ", clpDict['_verbose'] )
	print ("\n\n")
	
	return clpDict

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
