import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import scatterplot

pathlist = []


# Output image of clusters? 0 for no, 1 for yes:

cluster_image_output=1

# DBSCAN thresholds
eps_threshold=0.5
minimum_locs_threshold=15



# Any extra SR thresholds:
precision_threshold=100             # This is the upper precision threshold
SNR_threshold=1                      # Lowest SNB ratio allowed.    

Filename='Apt_FitResults.txt'   # This is the name of the SR file containing the localisations.

# Paths to analyse below:


pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Oligomer/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/2/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211117_asyn_tom20/Untreated/3/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211102_asyn_tom20/A53T/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/20211102_asyn_tom20/A53T/2/")


for path in pathlist:           # Go through each file
    print(path)    
    os.chdir(path)
    input_data = pd.read_table(Filename)       # Load the data
     
    
    df=input_data.copy()
    fit = df.drop('Source', 1)
    fit=fit.drop('SNR',1)
    
       
  
    Out_nc=open(path+'/'+'Apt_FitResults_withheader.txt','w')   # Open file for writing to.
        # Write the header of the file
    Out_nc.write("""#Localisation Results File
#FileVersion Text.D0.E0.V2
#Name Minee_Untreated_Tom_Apt1_1.tif (LSE)
#Source <gdsc.smlm.ij.IJImageSource><name>Minee_Untreated_Tom_Apt1_1.tif</name><width>428</width><height>684</height><frames>10000</frames><singleFrame>0</singleFrame><extraFrames>0</extraFrames><path>/Volumes/TOSHIBA EXT/SuperResData/20211103_Minee_Untreated/Minee_Untreated_Tom_Apt1_1.tif</path></gdsc.smlm.ij.IJImageSource>
#Bounds x0 y0 w428 h684
#Calibration <gdsc.smlm.results.Calibration><nmPerPixel>103.0</nmPerPixel><gain>2.17</gain><exposureTime>50.0</exposureTime><readNoise>0.0</readNoise><bias>0.0</bias><emCCD>false</emCCD><amplification>0.0</amplification></gdsc.smlm.results.Calibration>
#Configuration <gdsc.smlm.engine.FitEngineConfiguration><fitConfiguration><fitCriteria>LEAST_SQUARED_ERROR</fitCriteria><delta>1.0E-4</delta><initialAngle>0.0</initialAngle><initialSD0>2.0</initialSD0><initialSD1>2.0</initialSD1><computeDeviations>false</computeDeviations><fitSolver>LVM</fitSolver><minIterations>0</minIterations><maxIterations>20</maxIterations><significantDigits>5</significantDigits><fitFunction>CIRCULAR</fitFunction><flags>20</flags><backgroundFitting>true</backgroundFitting><notSignalFitting>false</notSignalFitting><coordinateShift>4.0</coordinateShift><shiftFactor>2.0</shiftFactor><fitRegion>0</fitRegion><coordinateOffset>0.5</coordinateOffset><signalThreshold>0.0</signalThreshold><signalStrength>30.0</signalStrength><minPhotons>0.0</minPhotons><precisionThreshold>10000.0</precisionThreshold><precisionUsingBackground>false</precisionUsingBackground><nmPerPixel>103.0</nmPerPixel><gain>2.17</gain><emCCD>false</emCCD><modelCamera>false</modelCamera><noise>0.0</noise><minWidthFactor>0.5</minWidthFactor><widthFactor>1.01</widthFactor><fitValidation>true</fitValidation><lambda>10.0</lambda><computeResiduals>false</computeResiduals><duplicateDistance>0.5</duplicateDistance><bias>0.0</bias><readNoise>0.0</readNoise><amplification>0.0</amplification><maxFunctionEvaluations>2000</maxFunctionEvaluations><searchMethod>POWELL_BOUNDED</searchMethod><gradientLineMinimisation>false</gradientLineMinimisation><relativeThreshold>1.0E-6</relativeThreshold><absoluteThreshold>1.0E-16</absoluteThreshold></fitConfiguration><search>2.5</search><border>1.0</border><fitting>3.0</fitting><failuresLimit>10</failuresLimit><includeNeighbours>true</includeNeighbours><neighbourHeightThreshold>0.3</neighbourHeightThreshold><residualsThreshold>1.0</residualsThreshold><noiseMethod>QUICK_RESIDUALS_LEAST_MEAN_OF_SQUARES</noiseMethod><dataFilterType>SINGLE</dataFilterType><smooth><double>0.5</double></smooth><dataFilter><gdsc.smlm.engine.DataFilter>MEAN</gdsc.smlm.engine.DataFilter></dataFilter></gdsc.smlm.engine.FitEngineConfiguration>
#Frame	origX	origY	origValue	Error	Noise	Background	Signal	Angle	X	Y	X SD	Y SD	Precision

""")
    Out_nc.write(fit.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)
    
    
    Out_nc.close() # Close the file.

    
    
    