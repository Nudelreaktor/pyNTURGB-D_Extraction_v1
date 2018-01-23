#!/usr/bin/env python2

import numpy as np

class single_hoj_set():

	def __init(self):
		self.__hoj_set_name = ""
		self.__hoj_set = np.array([])
		self.__hoj_labels = np.array([])

	def set_hoj_set( self, _hoj_set ):
		self.__hoj_set = _hoj_set

	def get_hoj_set( self ):
		return self.__hoj_set

	def set_hoj_label( self, _hoj_labels ):
		self.__hoj_labels = _hoj_labels

	def get_hoj_label( self ):
		return self.__hoj_labels

	def set_hoj_set_name( self, _name ):
		self.__hoj_set_name = _name

	def get_hoj_set_name( self ):
		return self.__hoj_set_name