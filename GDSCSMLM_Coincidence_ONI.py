#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:30:37 2021

@author: Mathew
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from skimage import filters,measure

 

# Settings
image_height=684
image_width=428
Pixel_size=117
scale=8
precision_threshold=250
eps_threshold=0.5
minimum_locs_threshold=15
prec_thresh=40

# Root path

root_path="/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/"

# GDSCSMLM file 1 - to perform cluster analysis on
filename_contains_ch1="Apt_FitResults.txt"      

# GDSCSMLM file 2 - to perform coincidence on.
filename_contains_ch2="modified_red_FitResults.txt"

# Folders to analyse:

pathlist=[]
# pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/1/")
# pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/2/")
# pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/3/")
# pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/1/")
# pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211102_asyn_tom20/A53T/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211102_asyn_tom20/A53T/2/")


# Various functions

#  Generate SR image (points)
def generate_SR(coords):
    SR_plot_def=np.zeros((image_height*scale,image_width*scale),dtype=float)
    j=0
    for i in coords:
        
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        # if(scale_xcoord<image_height and scale_ycoord<image_width):
        SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def SRGaussian(size, fwhm, center):

    sizex=size[0]
    sizey=size[1]
    x = np.arange(0, sizex, 1, float)
    y = x[0:sizey,np.newaxis]
    # y = x[:,np.newaxis]


    x0 = center[0]
    y0 = center[1]
    
    wx=fwhm[0]
    wy=fwhm[1]
    
    return np.exp(-0.5 * (np.square(x-x0)/np.square(wx) + np.square(y-y0)/np.square(wy)) )

def gkern(l,sigx,sigy):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx)/np.square(sigx) + np.square(yy)/np.square(sigy)) )
    # print(np.sum(kernel))
    # test=kernel/np.max(kernel)
    # print(test.max())
    return kernel/np.sum(kernel)

def generate_SR_prec(coords,precsx,precsy):
    box_size=20
    SR_prec_plot_def=np.zeros((image_height*scale,image_width*scale),dtype=float)
    dims=np.shape(SR_prec_plot_def)
    print(dims)
    j=0
    for i in coords:

      
        precisionx=precsx[j]/Pixel_size*scale
        precisiony=precsy[j]/Pixel_size*scale
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)
        scale_ycoord=round(ycoord*scale)
        
        
        
        sigmax=precisionx
        sigmay=precisiony
        
        
        # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        tempgauss=gkern(2*box_size,sigmax,sigmay)
        
        # SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        
        
        ybox_min=scale_ycoord-box_size
        ybox_max=scale_ycoord+box_size
        xbox_min=scale_xcoord-box_size
        xbox_max=scale_xcoord+box_size 
        
        
        if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
            SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
        
        
           
        j+=1
    
    return SR_prec_plot_def




# Perform DBSCAN on the coordinates. 

def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels

def generate_SR_cluster(coords,clusters):
    SR_plot_def=np.zeros((image_height*scale,image_width*scale),dtype=float)
    j=0
    for i in coords:
        if clusters[j]>-1:
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)
            scale_ycoord=round(ycoord*scale)
            # if(scale_xcoord<image_height and scale_ycoord<image_width):
            SR_plot_def[scale_ycoord,scale_xcoord]+=1
        
        j+=1
    return SR_plot_def

def generate_SR_prec_cluster(coords,precsx,precsy,clusters):
    box_size=50
    SR_prec_plot_def=np.zeros((image_height*scale+100,image_width*scale+100),dtype=float)
    SR_fwhm_plot_def=np.zeros((image_height*scale+100,image_width*scale+100),dtype=float)

    j=0
    for clu in clusters:
        if clu>-1:
       
            precisionx=precsx[j]/Pixel_size*scale
            precisiony=precsy[j]/Pixel_size*scale
            xcoord=coords[j,0]
            ycoord=coords[j,1]
            scale_xcoord=round(xcoord*scale)+50
            scale_ycoord=round(ycoord*scale)+50
            
            sigmax=precisionx
            sigmay=precisiony
            
            
            # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
            tempgauss=gkern(2*box_size,sigmax,sigmay)
            ybox_min=scale_ycoord-box_size
            ybox_max=scale_ycoord+box_size
            xbox_min=scale_xcoord-box_size
            xbox_max=scale_xcoord+box_size 
        
        
            if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
                SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
                
            tempfwhm_max=tempgauss.max()
            tempfwhm=tempgauss>(0.5*tempfwhm_max)
            
            tempfwhm_num=tempfwhm*(clu+1)
           
            
            if(np.shape(SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempfwhm)):
               plot_temp=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_add=np.zeros((2*box_size,2*box_size),dtype=float)
               plot_temp=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]
               plot_add_to=plot_temp==0
               
               plot_add1=plot_temp+tempfwhm_num
               
               plot_add=plot_add1*plot_add_to
               
               SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+plot_add
                
                
                # (SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm_num).where(SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]==0)
                # SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]=SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm
            
            # SR_tot_plot_def[SR_tot_plot_def==0]=1
            labelled=SR_fwhm_plot_def
            
            SR_prec_plot=SR_prec_plot_def[50:image_height*scale+50,50:image_width*scale+50]
            labelled=labelled[50:image_height*scale+50,50:image_width*scale+50]
            
            
        j+=1
    
    return SR_prec_plot,labelled,SR_fwhm_plot_def


def generate_SR_prec_FWHM(coords,precsx,precsy):
    box_size=50
    SR_prec_plot_def=np.zeros((image_height*scale+100,image_width*scale+100),dtype=float)
    SR_fwhm_plot_def=np.zeros((image_height*scale+100,image_width*scale+100),dtype=float)

    j=0
    for c in coords:
  
        precisionx=precsx[j]/Pixel_size*scale
        precisiony=precsy[j]/Pixel_size*scale
        xcoord=coords[j,0]
        ycoord=coords[j,1]
        scale_xcoord=round(xcoord*scale)+50
        scale_ycoord=round(ycoord*scale)+50
        
        sigmax=precisionx
        sigmay=precisiony
        
        
        # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        tempgauss=gkern(2*box_size,sigmax,sigmay)
        ybox_min=scale_ycoord-box_size
        ybox_max=scale_ycoord+box_size
        xbox_min=scale_xcoord-box_size
        xbox_max=scale_xcoord+box_size 
    
    
        if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
            SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
            
        tempfwhm_max=tempgauss.max()
        tempfwhm=tempgauss>(0.5*tempfwhm_max)
        
        tempfwhm_num=tempfwhm*1
       
        
        if(np.shape(SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempfwhm)):
           plot_temp=np.zeros((2*box_size,2*box_size),dtype=float)
           plot_add=np.zeros((2*box_size,2*box_size),dtype=float)
           plot_temp=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]
           plot_add_to=plot_temp==0
           
           plot_add1=plot_temp+tempfwhm_num
           
           plot_add=plot_add1*plot_add_to
           
           SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+plot_add
            
            
            # (SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm_num).where(SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]==0)
            # SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]=SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm
        
        # SR_tot_plot_def[SR_tot_plot_def==0]=1
        labelled=SR_fwhm_plot_def
        
        SR_prec_plot=SR_prec_plot_def[50:image_height*scale+50,50:image_width*scale+50]
        labelled=labelled[50:image_height*scale+50,50:image_width*scale+50]
            
            
        j+=1
    
    return SR_prec_plot,labelled




def analyse_labelled_image(labelled_image):
    
    measure_image=measure.regionprops_table(labelled_image,intensity_image=labelled_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

Output_all_cases = pd.DataFrame(columns=['Path','Number_of_clusters','Points_per_cluster_mean','Points_per_cluster_SD','Coincident','Coincident_fraction'])


for path in pathlist:
    print(path)
  
    # /////////////////////////////////////////////////////////////////////////////////////Channel 1//////////////////////////////////////////////////////////////////////////////////////////
    # Load the fits for Ch_1
    for root, dirs, files in os.walk(path):
                for name in files:
                        if filename_contains_ch1 in name:
                            
                
                                    resultsname = name
                                    print(resultsname)
                                    
    # This is the file to load for channel 1
    fits_path_ch1=path+resultsname
    loc_data_ch1=pd.read_table(fits_path_ch1)
    
    # Threshold some of the data (precision etc.)
    index_names = loc_data_ch1[loc_data_ch1['Precision (nm)']>prec_thresh].index
    loc_data_ch1.drop(index_names, inplace = True)
   

    # Extract useful data:
    coords_ch1= np.array(list(zip(loc_data_ch1['X'],loc_data_ch1['Y'])))
    precsx_ch1= np.array(loc_data_ch1['Precision (nm)'])
    precsy_ch1= np.array(loc_data_ch1['Precision (nm)'])
    xcoords_ch1=np.array(loc_data_ch1['X'])
    ycoords_ch1=np.array(loc_data_ch1['Y'])
    
    # Require one of dimensions for precisions (for GDSCSMLM, there's only one precision)
    precs_nm_ch1=precsx_ch1
    
    # Generate precision histogram.
    plt.hist(precs_nm_ch1, bins = 50,range=[0,100], rwidth=0.9,color='#ff0000')
    plt.xlabel('Precision (nm)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Localisation precision (channel 1)',size=20)
    plt.savefig(path+"Precision.pdf")
    plt.show()
        
    # Generate points SR (ESMB method):
    SR_ch1=generate_SR(coords_ch1)
    
    # Save the image
    imsr = Image.fromarray(SR_ch1)
    imsr.save(path+'SR_points_python_ch1.tif')
    
    # Generate precision image:
    SR_prec_ch1=generate_SR_prec(coords_ch1,precsx_ch1,precsy_ch1)
    
    # Save the image
    imsr = Image.fromarray(SR_prec_ch1)
    imsr.save(path+'SR_width_python_ch1.tif')
    
    
    # Perform DBSCAN cluster analysis
    clusters_ch1=cluster(coords_ch1)
    
    # Check how many localisations per cluster
 
    cluster_list_ch1=clusters_ch1.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function. 
    maximum=max(cluster_list_ch1)+1  
    
    
    cluster_contents_ch1=[]         # Make a list to store the number of clusters in
    
    for i in range(0,maximum):
        n=cluster_list_ch1.count(i)     # Count the number of times that the cluster number i is observed
       
        cluster_contents_ch1.append(n)  # Add to the list. 
    
    if len(cluster_contents_ch1)>0:
        average_locs_ch1=sum(cluster_contents_ch1)/len(cluster_contents_ch1)
 
        plt.hist(cluster_contents_ch1, bins = 20,range=[0,200], rwidth=0.9,color='#607c8e') # Plot a histogram. 
        plt.xlabel('Localisations per cluster')
        plt.ylabel('Number of clusters in CH1')
        plt.savefig(path+'Localisations.pdf')
        plt.show()
        
        cluster_arr_ch1=np.array(cluster_contents_ch1)
    
        median_locs_ch1=np.median(cluster_arr_ch1)
        mean_locs_ch1=cluster_arr_ch1.mean()
        std_locs_ch1=cluster_arr_ch1.std()
        
    
        # Generate the SR image.
        SR_Clu_ch1=generate_SR_cluster(coords_ch1,clusters_ch1)
        
        imsr = Image.fromarray(SR_Clu_ch1)
        imsr.save(path+'SR_points_python_clustered_ch1.tif')
        
        SR_clu_prec_ch1,labelled_ch1,SR_prec_plot_ch1=generate_SR_prec_cluster(coords_ch1,precsx_ch1,precsy_ch1,clusters_ch1)
        
        
        imsr = Image.fromarray(SR_clu_prec_ch1)
        imsr.save(path+'SR_width_python_clustered.tif')
        
        imsr = Image.fromarray(labelled_ch1)
        imsr.save(path+'SR_labelled_python_clustered_ch1.tif')
        
        labeltot_ch1=labelled_ch1.max()
        
        print('Total number of clusters in labelled image: %d'%labeltot_ch1)
    
 # /////////////////////////////////////////////////////////////////////////////////////Channel 2//////////////////////////////////////////////////////////////////////////////////////////
  # Load the fits for Ch_2
    for root, dirs, files in os.walk(path):
                for name in files:
                        if filename_contains_ch2 in name:
                            
                
                                    resultsname = name
                                    print(resultsname)
                                    
    # This is the file to load for channel 2
    fits_path_ch2=path+resultsname
    loc_data_ch2=pd.read_table(fits_path_ch2)
    
    # Threshold some of the data (precision etc.)
    index_names = loc_data_ch2[loc_data_ch2['Precision (nm)']>prec_thresh].index
    loc_data_ch2.drop(index_names, inplace = True)
   

    # Extract useful data:
    coords_ch2= np.array(list(zip(loc_data_ch2['X'],loc_data_ch2['Y'])))
    precsx_ch2= np.array(loc_data_ch2['Precision (nm)'])
    precsy_ch2= np.array(loc_data_ch2['Precision (nm)'])
    xcoords_ch2=np.array(loc_data_ch2['X'])
    ycoords_ch2=np.array(loc_data_ch2['Y'])
    
    # Require one of dimensions for precisions (for GDSCSMLM, there's only one precision)
    precs_nm_ch2=precsx_ch2
    
    # Generate precision histogram.
    plt.hist(precs_nm_ch2, bins = 50,range=[0,100], rwidth=0.9,color='#ff0000')
    plt.xlabel('Precision (nm)',size=20)
    plt.ylabel('Number of Features',size=20)
    plt.title('Localisation precision (channel 2)',size=20)
    plt.savefig(path+"Precision.pdf")
    plt.show()
        
    # Generate points SR (ESMB method):
    SR_ch2=generate_SR(coords_ch2)
    
    # Save the image
    imsr = Image.fromarray(SR_ch2)
    imsr.save(path+'SR_points_python_ch2.tif')
    
    # Generate precision image:
    SR_prec_ch2,SR_fwhm_ch2=generate_SR_prec_FWHM(coords_ch2,precsx_ch2,precsy_ch2)
    
    # Save the image
    imsr = Image.fromarray(SR_prec_ch2)
    imsr.save(path+'SR_precision_python_ch2.tif')
    
    
    imsr = Image.fromarray(SR_fwhm_ch2)
    imsr.save(path+'SR_width_python_ch2.tif')
    
    # Coincidence section:
    
    coincident_image=SR_fwhm_ch2*labelled_ch1
    
    # Look for labels:
        
    label_list, label_pixels = np.unique(coincident_image, return_counts=True)
    np.savetxt(path + 'Coinc_list.txt', label_list) 
    np.savetxt(path + 'Coinc_pixels.txt', label_pixels) 
    np.savetxt(path + 'Clister_size.txt', cluster_contents_ch1) 
    
    no_clust=len(cluster_contents_ch1)
    no_coinc=len(label_list)
    frac=no_coinc/no_clust
    
    Output_all_cases = Output_all_cases.append({'Path':path,'Number_of_clusters':no_clust,'Points_per_cluster_mean':mean_locs_ch1,'Points_per_cluster_SD':std_locs_ch1,'Coincident':no_coinc,'Coincident_fraction':frac},ignore_index=True)
        
    Output_all_cases.to_csv(root_path + 'GDSC_all_metrics_5.csv', sep = '\t')

