path= newArray(3);	


// All paths go below:

path[0]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/A53T/1/"
path[1]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/A53T/2/"
path[2]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/Untreated/"


for (i=0; i<path.length; i++){
dir=path[i];
to_open=dir+"Raw.tif";
run("Bio-Formats Importer", "open=[" + to_open + "] color_mode=Default crop rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT x_coordinate_1=0 y_coordinate_1=0 width_1=856 height_1=684");
					
makeRectangle(428, 0, 856, 684);
run("Duplicate...", "duplicate range=1-10000");
saveAs("Tiff", dir+"tom20.tif");
close();
makeRectangle(0, 0, 428, 684);
run("Duplicate...", "duplicate range=10000-15000");
saveAs("Tiff", dir+"apt.tif");
close();
close();
	
}

