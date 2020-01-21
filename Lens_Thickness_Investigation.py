# -*- coding: utf-8 -*-

import raytracer as rt
import numpy as np
import matplotlib.pyplot as plt

csfont = {'family':'serif'}

lens1 = rt.ThickLens(1/50, -1/50, 10)
lens2 = rt.ThickLens(1/25, -1/25, 10)
lens3 = rt.ThickLens(1/10, -1/10, 10)

f_1 = lens1.plot_f_d(2, 100,0.5)
RMS_1 = lens1.plot_RMS_d(2, 100,0.5)
f_2 = lens2.plot_f_d(2, 100, 0.5)
RMS_2 = lens2.plot_RMS_d(2, 100, 0.5)
f_3 = lens3.plot_f_d(2,100, 0.5)
RMS_3 = lens3.plot_RMS_d(2, 100, 0.5)

fig1 = plt.figure(1)
plt.grid()
plt.rcParams['figure.figsize'] = [8,6]
plt.xlabel("Thickness (mm)", fontsize = 15,**csfont)
plt.ylabel("Focal length (mm)", fontsize = 15,**csfont)
plt.plot(f_1[0], f_1[1], color = "Indigo", label = "c = [0, -0.02]")
plt.plot(f_2[0], f_2[1], color = "dodgerblue", label = "c = [0, -0.04]")
plt.plot(f_3[0], f_3[1], color = "black", label = "c = [0, -0.1]")
plt.legend(fontsize = 'large')


fig2 = plt.figure(2)
plt.grid()
plt.rcParams['figure.figsize'] = [8,6]
plt.xlabel("Thickness (mm)", fontsize = 15,**csfont)
plt.ylabel("RMS Spot Radius at Paraxial Focus (mm)", fontsize = 13,**csfont)
plt.plot(RMS_1[0], RMS_1[1], color = "indigo", label = "c = [0, -0.02]")
plt.plot(RMS_2[0], RMS_2[1], color = "dodgerblue", label = "c = [0, -0.04]")
plt.plot(RMS_3[0], RMS_3[1], color = "black", label = "c = [0, -0.1]")
plt.legend(fontsize = 'large')

print(lens1.getRMS3mm())
print(lens2.getRMS3mm())
print(lens3.getRMS3mm())
print(lens1.getRMS25mm())
print(lens2.getRMS25mm())
print(lens3.getRMS25mm())


#f1 = lens1.plot_f_opt_d(2, 100, 1)
#f2 = lens2.plot_f_opt_d(2, 100, 1)
#
#fig3 = plt.figure(3)
#plt.grid()
#plt.xlabel("Thickness (mm)", fontsize = 15,**csfont)
#plt.ylabel("Circle of Least Confusion Position (mm)", fontsize = 15,**csfont)
#plt.plot(f1[0], f1[1], color = "Indigo", label = "c = [0.02, -0.02]")
#plt.plot(f2[0], f2[1], 
#         color = "dodgerblue", label = "c = [0.04, -0.04]")
#plt.plot(lens3.plot_f_opt_d(2, 100, 1)[0], lens3.plot_f_opt_d(2, 100, 1)[1], 
#         color = "black", label = "c = [0.1, -0.1]")
#plt.legend()
#
#fig4 = plt.figure(4)
#plt.grid()
#plt.xlabel("Thickness (mm)", fontsize = 15,**csfont)
#plt.ylabel("RMS Spot Radius at COLC (mm)", fontsize = 15,**csfont)
#plt.plot(lens1.plot_RMS_opt_d(2, 100, 1)[0], lens2.plot_RMS_opt_d(2, 100, 1)[1], 
#         color = "Indigo", label = "c = [0.02, -0.02]")
#plt.plot(lens2.plot_RMS_opt_d(2, 100, 1)[0], lens2.plot_RMS_opt_d(2, 100, 1)[1], 
#         color = "dodgerblue", label = "c = [0.04, -0.04]")
#plt.plot(lens3.plot_RMS_opt_d(2, 100, 1)[0], lens3.plot_RMS_opt_d(2, 100, 1)[1], 
#         color = "black", label = "c = [0.1, -0.1]")
#plt.legend()
