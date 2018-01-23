#!/usr/bin/env/python
# coding : utf-8

# Python module import
import sys
import math
import numpy as np

# Ppe module import
import joint as j
import frameHeader as fHeader
 
class data_extraction(): 

	def extract_data( _self, _all_skeleton_frames, _extraction_field, _sample_strategy="random", _number_of_subframes=8, _verbose=False ):

		_extracted_data = []

		for _frame in _all_skeleton_frames:

			_joint_list = _frame.get_ListOfJoints()
			_needed_frame_joint_data = []

			for _joint in _joint_list:

				if _extraction_field is 'world':
					_needed_frame_joint_data.append(_joint.get_WorldJoint())


			_extracted_data.append(_needed_frame_joint_data )

		# print("Len org data: %10d" %sys.getsizeof(_all_skeleton_frames))
		# print("Len new data: %10d" %sys.getsizeof(_extracted_data))

		# print("org data: ", _all_skeleton_frames)
		# print("new data: ", _extracted_data)

		return _extracted_data

	def frame_subsampling( _self, _set=[], _sample_strategy="random", _number_of_subframes=8, _verbose=False ):

		_final_data_after_subsampling = []

		# Get some informations about the number of frames in the actual set.
		number_of_frames = len(_set)

		# Compute the size of the 8 buckets depending of the number of frames of the set.
		bucket_size = math.floor( number_of_frames / _number_of_subframes )
		remain = number_of_frames - ( bucket_size * _number_of_subframes )
		gap = math.floor(remain / 2.0)

		if( _verbose == True ):
			print("CTSVM :: Number of frames: ", number_of_frames)
			print("CTSVM :: Bucket Size: ", bucket_size)

		# Take a random frame from each bucket and store it as array entry in the _svm_structure ( 8 per )
		for k in range(0,_number_of_subframes):

			# Choose the sampling strategy
			# Random frame per bucket
			if( _sample_strategy == "random"):
				# Get the random frame -> randint(k(BS),k+1(BS)) ==> k-1(B) < randomInt < k(B)
				random_frame_number = random.randint((gap+(k*bucket_size)),(gap+((k+1)*bucket_size)) )
			# First frame per bucket
			elif( _sample_strategy == "first"):
				random_frame_number = int(gap+(k*bucket_size)+1)
			# Mid frame per bucket
			elif( _sample_strategy == "mid"):
				random_frame_number = int(gap+(k*bucket_size)+int(math.floor(bucket_size/2)))
			# Last frame per bucket
			elif( _sample_strategy == "last"):
				random_frame_number = int(gap+(k*bucket_size)+bucket_size)

			if( _verbose == True ):
				print("CTSVM :: Lower border: "+str(gap+(k*bucket_size)))
				print("CTSVM :: Upper border: "+str(gap+((k+1)*bucket_size)))
				print("CTSVM :: Lower and upper border gap: "+str(gap))
				print(random_frame_number)

			# Convert the frame to the svm structure 
			# Get the random frame and the corresponding label
			if( random_frame_number > 0 ):
				# Collect the data from the 8 buckets in a list.
				_final_data_after_subsampling.append(_set[random_frame_number-1]);
			else:
				# Collect the data from the 8 buckets in a list.
				_final_data_after_subsampling.append(_set[random_frame_number]);

		return _final_data_after_subsampling