#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to do basic PIV and send it to myPTV tracking

"""
import workflow_HW as wf
from coarsePIV import runCoarsePIV   
import matplotlib as plt
        
directory = '/Users/hannahwalker/Desktop/MyPTV-master/Experiment/'
param_file = directory + 'params_file_TEST.yml'
action = 'trackingMF'

time, MF = runCoarsePIV() # get a mean flow for data chunk using PIV
    
#mean_flow = array([0.0, 0.0, 0.0])
Fr = 65 
mean_flow = MF*(1/Fr) # get mean flow in lab coordinates relative to frame rate

plt.figure(1)
plt.plot(time,mean_flow)

work = wf.workflow_HW(param_file, action, mean_flow)
