path= newArray(3);	


// All paths go below:

path[0]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/A53T/1/"
path[1]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/A53T/2/"
path[2]="/Volumes/VERBATIM HD/Data_For_Mathew/Minee Oligomer and TOM20/Untreated/"


for (i=0; i<path.length; i++){
dir=path[i];


stem="pos_Z0.tif";
processFiles(dir,stem);

stem="pos_Z0_10.tif";
processFiles(dir,stem);


///////////// The code below is what is run on each image that's opened.

run("Concatenate...", "all_open title=[Concatenated Stacks]");		// Join all of the images together

		
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

function processFiles(dir,stem) 

	{

	list = getFileList(dir);

		for (i=0; i<list.length; i++)

		 {

		if (!startsWith(list[i],"Log"))

			{

			if (endsWith(list[i], "/"))

		              processFiles(""+dir+list[i]);

         			 else 

			{

		             showProgress(n++, count);

            			path = dir+list[i];

	            		processFile(path,stem);

			}

			}

		}

	}

function processFile(path,stem) 

	{
		       	if (endsWith(path, stem) ) 
 
		{
			
			open(path);
			


		}
	}