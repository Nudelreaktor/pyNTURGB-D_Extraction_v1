# pyNTU_hoj_gen_v1

We refactored the hoj_gen data extraction tool for the Rose Lab NTURGB-D dataset by Sharoudy et.al. ( https://github.com/shahroudy/NTURGB-D ) in the last few weeks.
And we've changed a lot.
You should read the description below to get informed about all changes.
And to start the information tour properly we will begin with the requirements followed by the known bugs and memory recomendations.

# Requirments #

-> Install a python ( >= 2.7 )

-> Install numpy 

-> Install shutil 

-> Get the dataset http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp ( only the skeleton data )

-> Clone/Download this repository

# Occlusions # 

We can't compute data frames where body parts are missing ( occlusion ).
This is a real problem. But the problem is in the dataset not in the data transformation tool.
It is that the basic data contain the joints of a skeleton frame as a number of lines where the line numbers define the joint numbers.
But if a occlusion occured, the correlation between line number and joint number isn't given anymore.
So if this happen, we have to decide what we will do.

	-> We could skip the entire set.
	But this means that we will lose data. ( Again. Remember the missing_skeleton file in the conf folder.)

	-> Otherwise we could skip just the frame(s).
	Yes. Frames. More likely sequences of frames.
	This leads to the following situation.
	If we lost bigger parts of a set how useful is it?
	So this strategy means we maybe get corrupted data where a person will do ultra fast postion and/or orientation switches.

We have implemented both strategies.
But the basic problems still exists. 
We can't deal with occluded body parts ( what means, we can't process data where an arm is missing or something similar. ).	

# Main Memory Recommendations #

There are ~56600 sets a ~100 frames in the dataset and we can't store them all in the memory during the computation.
Therefor, we will devide the dataset in 4 temporary parts ( stored in tmp_data/ ) during the first step of the hoj computation ( read from the original files ). 
( Later, we will implement a command line flag where you can specify if the data set should be parted and if so how many parts you want. )
After that, we will load the data parts into memory to compute the hoj'es out of the tmp files and to store them in a final data container in the pickles folder.

This means two things:

1) You don't have to compute the complete data set again if the hoj computation ( step 2 in the chain ) fails at some point. 
	You have the temporary parts still in the tmp_parts folder. 

2) We need at least a minimal amount of memory for the procedure.
	First of all, it's possible to compute the full chain with 8 Gb of Ram, but we don't recommend it.
	If this will be successful depends strongly on the systems memory handling strategy and on the task(s) you will do besides the dataset computation. 
	( Coffee would be a good idea because coffee is always a good idea. )
	To prevent some lost hours of computation time without a sufficient result we've defined minimal specifications.

	We recommend: 
  
		-> Minimum:      8 Gb ( If you need just parts of the full set or you have a lot of time. Approx: 4 hours for the full set. )
		-> Sufficient:  12 Gb ( Yep, it works and i can play minesweeper. )
		-> Recommended: 16 Gb ( For full performance. Testcase: Some rounds PUBG + hoj_gen full set computation. )
