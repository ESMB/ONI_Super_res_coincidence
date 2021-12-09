path= newArray(1);	


// All paths go below:


path[0]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/WT/1/"

for (i=0; i<path.length; i++){
dir=path[i];
to_open=dir+"apt.tif";

config_file="/Users/Mathew/gdsc.smlm.settings.xml";			// This is where the GDSCSMLM cofig. file is saved. This can be found by running the fit peaks command. 
pixel_size=103;			// Size of each pixel in nm
exposure=50;			// Exposure time
precision_thresh=20;	// Precision threshold.
signal_thresh=30;		// Signal threshold.

open(to_open);

run("Peak Fit", "template=[None] config_file="+config_file+" calibration="+pixel_size+" gain=2.17 exposure_time="+exposure+" initial_stddev0=2.000 initial_stddev1=2.000 initial_angle=0.000 spot_filter_type=Single spot_filter=Mean smoothing=0.50 search_width=2.50 border=1 fitting_width=3 fit_solver=[Least Squares Estimator (LSE)] fit_function=Circular fail_limit=10 include_neighbours neighbour_height=0.30 residuals_threshold=1 duplicate_distance=0.50 shift_factor=2 signal_strength="+signal_thresh+" min_photons=0 min_width_factor=0.50 width_factor=2 precision="+precision_thresh+" results_table=Uncalibrated image=[Localisations (width=precision)] image_precision=5 image_scale=8 results_dir="+dir+" results_in_memory camera_bias=0 fit_criteria=[Least-squared error] significant_digits=5 coord_delta=0.0001 lambda=10.0000 max_iterations=20 stack");
saveAs("Tiff", dir+"Apt_super_res.tif");
close();
selectWindow("Fit Results");
saveAs("text",dir+"Apt_FitResults.txt");
run("Close");
close();

to_open=dir+"tom20.tif";
open(to_open);

run("Peak Fit", "template=[None] config_file="+config_file+" calibration="+pixel_size+" gain=2.17 exposure_time="+exposure+" initial_stddev0=2.000 initial_stddev1=2.000 initial_angle=0.000 spot_filter_type=Single spot_filter=Mean smoothing=0.50 search_width=2.50 border=1 fitting_width=3 fit_solver=[Least Squares Estimator (LSE)] fit_function=Circular fail_limit=10 include_neighbours neighbour_height=0.30 residuals_threshold=1 duplicate_distance=0.50 shift_factor=2 signal_strength="+signal_thresh+" min_photons=0 min_width_factor=0.50 width_factor=2 precision="+precision_thresh+" results_table=Uncalibrated image=[Localisations (width=precision)] image_precision=5 image_scale=8 results_dir="+dir+" results_in_memory camera_bias=0 fit_criteria=[Least-squared error] significant_digits=5 coord_delta=0.0001 lambda=10.0000 max_iterations=20 stack");
saveAs("Tiff", dir+"tom20_super_res.tif");
close();
selectWindow("Fit Results");
saveAs("text",dir+"tom20_FitResults.txt");
run("Close");
close();
	
}

