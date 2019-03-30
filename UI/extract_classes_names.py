#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:26:22 2019

@author: Samuel_Levesque
"""

from os import walk

data_folder = r"~/Desktop/Travaux école/Université/Maîtrise en AI/Hiver 2019/Deep Learning/Projet de session/data/train_simplified"

f = []
for (dirpath, dirnames, filenames) in walk(data_folder):
    f.extend(filenames)
    break
