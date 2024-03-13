## Download Quantitative Phantom
We have downloaded the T1/T2/PD maps. They were obtained from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1). To save you the hassle of converting the .mnc files to numpy, you can download the numpy files directly by visiting the folowing google drive link:    
https://drive.google.com/drive/folders/1aLT0-Hf38MKXpCbHCXpO-pVsPPGGqB0z?usp=drive_link


There are two files in the google drive folder:   
- `brainweb_phantom.npz` containing the t1/t2/pd maps, t1 and t2 maps are in milliseconds, pd is scaled from 0 to 1, and each of the maps has a matrix size of (180, 216, 180)
- `mrf_sequence.mat` containing a train of flip angles and TRs used for an MRF acquisiton. The trs are in milliseconds, and flip angles in radians.
