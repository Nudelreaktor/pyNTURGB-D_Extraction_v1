#!/usr/bin/env python3

# Python module import
import numpy as np
import math as ma
import time
import copy as cp
import warnings

from sklearn.metrics.pairwise import euclidean_distances

# Ppe module import
import frameHeader
import joint

#########################################################################################
##
##  Get the joint wise distance descriptor
##
#########################################################################################
#def get_distance_descriptor(list_of_joints, reference_join, reference_join_up, reference_join_left, reference_join_right, joint_indexes = [], n_time = 0.0):
def get_distance_descriptor(list_of_joints, joint_indexes = [], n_time = 0.0, local=False):
	
	t0 = time.time()

	# get joints to compute
	if(joint_indexes):
		joints_to_compute = []
		for index in joint_indexes:
			joints_to_compute.append(cp.deepcopy(list_of_joints[index]))
	else:
		joints_to_compute = list_of_joints
	
	# Compute a local skeleton ( reference joint is the mid between hipl and hipr )
	if local is True:
		#translation
		translation_vector = np.array([-reference_join.get_WorldJoint()[0], -reference_join.get_WorldJoint()[1], -reference_join.get_WorldJoint()[2]])

		# print(translation_vector)

		for joint in joints_to_compute:
			point = np.array(joint.get_WorldJoint())
			transformed_point = point + translation_vector
			joint.set_WorldJoint(transformed_point.item(0),transformed_point.item(1),transformed_point.item(2))

	# Compute output vector.
	coords = np.zeros(len(joints_to_compute) * ( len(joints_to_compute) ) )
	
	# Compute joint to joint distance descriptor.
	# Run through all joints
	for idx_host,joint_host in enumerate(joints_to_compute):

		host_vec = np.array(joint_host.get_WorldJoint())

		# Compute for each joint ( except with itself ) a descriptor entry 
		# containing euclidean distances to each other joint. ( TODO Later followed by an oriantation ??? )
		# n joints x n-1 distances = ((n^2)-n) view independend entrys
		guest_index = 0
		for idx_guest,joint_guest in enumerate(joints_to_compute):

			# # As long as host and guest are different entities.
			# if idx_host != idx_guest:

			guest_vec = np.array(joint_guest.get_WorldJoint())

			# Compute euclidean between host and guest.
			dist = np.linalg.norm( host_vec - guest_vec )

			# Store joint to joint distance.
			coords[idx_host * ( len(joints_to_compute) -1 ) + guest_index] = dist
			guest_index = guest_index + 1 

			# TODO Include local transformation later if necessary.
			# x,y,z = transform_coordinate_linear(x_vector,y_vector,z_vector,joint.get_WorldJoint())
		
	




	t1 = time.time()
	n_time += t1 - t0
	
	return coords,n_time


#########################################################################################
##
##     get the 3D coordinates
##
#########################################################################################
def get_joint_coordinates(list_of_joints, reference_join, reference_join_up, reference_join_left, reference_join_right, joint_indexes = [], n_time = 0.0, local=False):
	
	t0 = time.time()
	
	# get joints to compute
	if(joint_indexes):
		joints_to_compute = []
		for index in joint_indexes:
			joints_to_compute.append(cp.deepcopy(list_of_joints[index]))
	else:
		joints_to_compute = list_of_joints
		
	coords = np.zeros(len(joints_to_compute) * 3)
	
	if local is False:
		for idx,joint in enumerate(joints_to_compute):
			coords[idx*3] = joint.get_WorldJoint()[0]
			coords[(idx*3)+1] = joint.get_WorldJoint()[1]
			coords[(idx*3)+2] = joint.get_WorldJoint()[2]
	else:
	
		#translation
		translation_vector = np.array([-reference_join.get_WorldJoint()[0], -reference_join.get_WorldJoint()[1], -reference_join.get_WorldJoint()[2]])

		# print(translation_vector)

		for joint in joints_to_compute:
			point = np.array(joint.get_WorldJoint())
			transformed_point = point + translation_vector
			joint.set_WorldJoint(transformed_point.item(0),transformed_point.item(1),transformed_point.item(2))
		
		#
		# X = right_hip to left_hip
		#
		# Y = bottom_center to spine
		# 		(allmost perpendicular to alpha)
		#
		x_vector = np.array(reference_join_left.get_WorldJoint()) - np.array(reference_join_right.get_WorldJoint())
		y_vector = np.array(reference_join_up.get_WorldJoint()) - np.array(reference_join.get_WorldJoint())

		# calculate z
		z_vector = np.cross(x_vector,y_vector)

		# recalculate y so it is perpendicular to XZ plane
		y_vector = np.cross(z_vector,x_vector)
		
		for idx,joint in enumerate(joints_to_compute):
			x,y,z = transform_coordinate_linear(x_vector,y_vector,z_vector,joint.get_WorldJoint())
			coords[idx*3] = x
			coords[(idx*3)+1] = y
			coords[(idx*3)+2] = z
	
	
	t1 = time.time()
	n_time += t1 - t0
	
	return coords,n_time

#########################################################################################
##
##		compute the HoJ3D
##
#########################################################################################
def compute_hoj3d( list_of_joints, reference_join, reference_join_up, reference_join_left, reference_join_right, joint_indexes = [], use_triangle_function = False, n_time = 0.0, _depth_measurement = None, _body_parts = False):

	t0 = time.time()

	# bins for alpha, the horizontal angle, starting from 
	alpha_bin = [0,30,60,90,120,150,180,210,240,270,300,330,360]
	# bins tor theta, the vertical angle, starting from north pole
	theta_bin = [-15,15,45,75,105,135,165,195]

	# the historamm of joints 3D
	hoj3d = np.zeros((7,12))
	if _depth_measurement is not None:
		hoj3d = np.zeros((7,24))

	# get joints to compute
	if(joint_indexes):
		joints_to_compute = []
		for index in joint_indexes:
			joints_to_compute.append(cp.deepcopy(list_of_joints[index]))
	else:
		joints_to_compute = list_of_joints
	
	# assign probability function
	probability_function = p_function
	if(use_triangle_function):
		probability_function = trinangle_function
		
	# assign hoj add function
	hoj_add_function = add_to_hoj_without_depth
	if _depth_measurement is not None:
		hoj_add_function = add_to_hoj_with_depth

	# calculate bodyparts
	arm_length, left_hand_sholder, right_hand_sholder = calculate_arm_length(list_of_joints)
	leg_length, left_foot_hip, right_foot_hip = calculate_leg_length(list_of_joints)
		

	# calculate radius for the inner cylinder/sphere
	cut_radius = arm_length
	# print('cut_radius='+str(cut_radius))

	#translation
	translation_vector = np.array([-reference_join.get_WorldJoint()[0], -reference_join.get_WorldJoint()[1], -reference_join.get_WorldJoint()[2]])

	# print(translation_vector)

	for joint in joints_to_compute:
		point = np.array(joint.get_WorldJoint())
		transformed_point = point + translation_vector
		joint.set_WorldJoint(transformed_point.item(0),transformed_point.item(1),transformed_point.item(2))
	
	#
	# X = right_hip to left_hip
	#
	# Y = bottom_center to spine
	# 		(allmost perpendicular to alpha)
	#
	x_vector = np.array(reference_join_left.get_WorldJoint()) - np.array(reference_join_right.get_WorldJoint())
	y_vector = np.array(reference_join_up.get_WorldJoint()) - np.array(reference_join.get_WorldJoint())

	# calculate z
	z_vector = np.cross(x_vector,y_vector)

	# recalculate y so it is perpendicular to XZ plane
	y_vector = np.cross(z_vector,x_vector)

	for joint in joints_to_compute:

		# Catch the zero division raised by NaNs in the skeleton files
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			try:
				r,flat_r,alpha,theta = transform_coordinate_shpere(x_vector,y_vector,z_vector,joint.get_WorldJoint())
			except Warning:
				return [],-1

		considered_radius = flat_r
		if _depth_measurement is not "cylinder":
			considered_radius = r
		
		inner, outer = r_function(considered_radius, cut_radius)

		j = 0
		for t in theta_bin:
			if(t >= 180):
				break

			# find theta-bins to calculate
			if((theta - t > 60) or (theta - t < -30)):
				j+=1
				continue

			i = 0
			for a in alpha_bin:
				if(a == 360):
					break

				# find alpha-bins to calculate
				if((alpha - a <= 60) and (alpha - a >= -30)):
					probability = abs((probability_function(alpha,(alpha_bin[i+1]+alpha_bin[i])/2))) * abs((probability_function(theta,(theta_bin[j+1]+theta_bin[j])/2)))

					# print(probability)
					hoj3d = hoj_add_function(hoj3d, probability, inner, outer, i, j)

				# wrap around the sphere
				if((alpha+360 - a <= 60) or (alpha-360 - a >= -30)):
					if(alpha < 30):
						probability = abs((probability_function(alpha+360,(alpha_bin[i+1]+alpha_bin[i])/2))) * abs((probability_function(theta,(theta_bin[j+1]+theta_bin[j])/2)))

						# print(probability)
						hoj3d = hoj_add_function(hoj3d, probability, inner, outer, i, j)

					elif(alpha > 330):
						probability = abs((probability_function(alpha-360,(alpha_bin[i+1]+alpha_bin[i])/2))) * abs((probability_function(theta,(theta_bin[j+1]+theta_bin[j])/2)))
						
						# print(probability)
						hoj3d = hoj_add_function(hoj3d, probability, inner, outer, i, j)

				i += 1
			j += 1

	# debug print
	#np.set_printoptions(precision = 3,suppress = True)
	#print(np.array(hoj3d))

	#hoj = np.array(hoj3d)
	#print(hoj.sum())
	
	# add bodyparts
	if _body_parts:
		hoj3d = np.append(hoj3d.flatten(),[left_hand_sholder, right_hand_sholder, left_foot_hip, right_foot_hip])
	
	# Flat the data. Always.
	hoj3d = hoj3d.flatten()

	t1 = time.time()
	n_time += t1 - t0
	
	return hoj3d,n_time

def transform_coordinate_shpere(x,y,z,vector):
	vector = np.array(vector)
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	# for alpha:
	# project vector on ground plane
	# 
	# vector_p = vector_v + u * vector_y    => normal on the plane
	#
	# 0 = [vector_p - vector_x] * vector_y  => plane in normal form
	#
	# => 0 = [vector_v + u * vector_y - vector_x] * vector_y
	# => 0 = vector_v * vector_y + u * vector_y * vector_y - vector_x * vector_y
	# => u = (vector_x * vector_y - vector_v * vector_y) / (vector_y * vector_y)

	u = (np.dot(x,y) - np.dot(vector,y)) / np.dot(y,y)

	projected_vector = vector + u * y

#	print(projected_vector)

	r = ma.sqrt((vector * vector).sum())
	projected_r = ma.sqrt((projected_vector * projected_vector).sum())
	x_len = ma.sqrt((x * x).sum())
	y_len = ma.sqrt((y * y).sum())
	z_len = ma.sqrt((z * z).sum())

	cos_alpha = np.dot(x,vector) / (projected_r * x_len)
	cos_beta  = np.dot(z,vector) / (projected_r * z_len)
	cos_theta = np.dot(y,vector) / (r * y_len)

	# print(cos_alpha)
	if( cos_alpha < -1.0 ):
		cos_alpha = -1.0

	if( cos_alpha > 1.0 ):
		cos_alpha = 1.0

	alpha = 0.0
	alpha = ma.acos(cos_alpha) * 360 / (2 * ma.pi)

	if(cos_beta < 0):			# if vector is to the right of x
		alpha = 360 - alpha

	theta = ma.acos(cos_theta) * 360 / (2 * ma.pi)

	if(ma.isnan(alpha)):
		alpha = 0
	if(ma.isnan(theta)):
		theta = 0

	return r, projected_r, alpha, theta
	
def transform_coordinate_linear(x,y,z,vector):
	vector = np.array(vector)
	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	x_len = vector_len(x)
	y_len = vector_len(y)
	z_len = vector_len(z)

	# old_vector = (x,y,z)
	# new_vector = (s,t,u)

	s_vector = ((vector * x) / (x_len * x_len)) * x
	t_vector = ((vector * y) / (y_len * y_len)) * y
	u_vector = ((vector * z) / (z_len * z_len)) * z
	
	s = vector_len(s_vector)
	t = vector_len(t_vector)
	u = vector_len(u_vector)
	
	
	# check if vector os in the same direction as the axis
	if (vector_len(s_vector + x) < vector_len(s_vector) or vector_len(s_vector + x) < vector_len(x)):
		s = -s;
	if (vector_len(t_vector + y) < vector_len(t_vector) or vector_len(t_vector + y) < vector_len(y)):
		t = -t;
	if (vector_len(u_vector + z) < vector_len(u_vector) or vector_len(u_vector + z) < vector_len(z)):
		u = -u;

	return s,t,u

def p_function(x,my):

	# original
	# p = (1 / (ma.sqrt(2 * ma.pi))) * ma.pow(ma.e,(-0.5 * (x-my) * (x-my)))
	# changed
	p = ma.exp(-0.005 * (x-my) * (x-my))

	#print(x)
	#print(my)
	#print(p)

	return p

def trinangle_function(x,my):

	if(x < my):
		p = (x - my + 30) / 30
	else:
		p = (-x + my + 30) / 30

	return max(p,0)

# calculate probability using radius
def r_function(x, cut_radius = 0.15):
	outer = 0 if x<cut_radius else 1
	inner = 1 - outer

	return inner,outer

def calculate_arm_length(list_of_joints):
	# calculate arm lenght
	#			sholder								elbow
	left_upper_arm = np.array(list_of_joints[4].get_WorldJoint()) - np.array(list_of_joints[5].get_WorldJoint())
	#			elbow								hand
	left_under_arm = np.array(list_of_joints[5].get_WorldJoint()) - np.array(list_of_joints[6].get_WorldJoint())
	left_arm = ma.sqrt((left_upper_arm * left_upper_arm).sum()) + ma.sqrt((left_under_arm * left_under_arm).sum())

	#			sholder								elbow
	right_upper_arm = np.array(list_of_joints[8].get_WorldJoint()) - np.array(list_of_joints[9].get_WorldJoint())
	#			elbow								hand
	right_under_arm = np.array(list_of_joints[9].get_WorldJoint()) - np.array(list_of_joints[10].get_WorldJoint())
	right_arm = ma.sqrt((right_upper_arm * right_upper_arm).sum()) + ma.sqrt((right_under_arm * right_under_arm).sum())
	arm_length = ((left_arm + right_arm) / 2)

	# calculate distance sholder -> hand
	#			sholder								hand
	left_vector = np.array(list_of_joints[4].get_WorldJoint()) - np.array(list_of_joints[6].get_WorldJoint())
	distance_left = ma.sqrt((left_vector * left_vector).sum())
	# nomalize
	left = distance_left / arm_length

	#			sholder								hand
	right_vector = np.array(list_of_joints[8].get_WorldJoint()) - np.array(list_of_joints[10].get_WorldJoint())
	distance_right = ma.sqrt((right_vector * right_vector).sum())
	# normalize
	right = distance_right / arm_length

	return arm_length, left, right

def calculate_leg_length(list_of_joints):
	# calculate leg lenght
	#			hip								knee
	left_upper_leg = np.array(list_of_joints[12].get_WorldJoint()) - np.array(list_of_joints[13].get_WorldJoint())
	#			knee								foot
	left_under_leg = np.array(list_of_joints[13].get_WorldJoint()) - np.array(list_of_joints[14].get_WorldJoint())
	left_leg = ma.sqrt((left_upper_leg * left_upper_leg).sum()) + ma.sqrt((left_under_leg * left_under_leg).sum())

	#			hip								knee
	right_upper_leg = np.array(list_of_joints[16].get_WorldJoint()) - np.array(list_of_joints[17].get_WorldJoint())
	#			knee								foot
	right_under_leg = np.array(list_of_joints[17].get_WorldJoint()) - np.array(list_of_joints[18].get_WorldJoint())
	right_leg = ma.sqrt((right_upper_leg * right_upper_leg).sum()) + ma.sqrt((right_under_leg * right_under_leg).sum())
	leg_length = ((left_leg + right_leg) / 2)

	# calculate distance hip -> foot
	#			hip								foot
	left_vector = np.array(list_of_joints[12].get_WorldJoint()) - np.array(list_of_joints[14].get_WorldJoint())
	distance_left = ma.sqrt((left_vector * left_vector).sum())
	# nomalize
	left = distance_left / leg_length

	#			hip								foot
	right_vector = np.array(list_of_joints[16].get_WorldJoint()) - np.array(list_of_joints[18].get_WorldJoint())
	distance_right = ma.sqrt((right_vector * right_vector).sum())
	# normalize
	right = distance_right / leg_length

	return leg_length, left, right
	

# helper function
def add_to_hoj_with_depth(hoj3d, probability, inner_propability, outer_propability, i, j):
	# print(probability)
	hoj3d[j][(i * 2)] += probability * inner_propability
	hoj3d[j][(i * 2)+1] += probability * outer_propability
	return hoj3d
	
def add_to_hoj_without_depth(hoj3d, probability, inner_propability, outer_propability, i, j):
	hoj3d[j][i] += probability
	return hoj3d

def vector_len(vector):
	return (np.sqrt((vector * vector).sum()))