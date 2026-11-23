# MultiCompartmentModel_Python - How do

1) [position and diameter Exel file]
In Matlab: (only done once, so that we have the 2 csv files in the Python ordner)
   load('matlab_and_excel_data.mat');
   writetable(positionData, 'positionData.csv')
   writetable(diameterData, 'diameterData.csv')

   Shift the two files to the Python project ordner.

2) [branchData mat file]
In Matlab/astro_geometry.m:
   Set wanted cellID in line 43 and execute 	branchData=astro_geometry(); in Command Window

   save branchData.mat branchData;

   Shift mat file to Python project ordner

3) [morph.ini file]
In SingleCellModel_Python\output\morph.ini:
  Set the right cellID, maximum_level, smallest_diameter,   radius_soma, compartment_time_curves_to_plot

4) [Use morph model for simulation]
In SingleCellModel_Python\main.py:
   Line31: simulate_morpology_model should be set to True, so 	that morphology model is used for simulation
   Line87: set save_Data_in_csv to True if you want to save 	for each compartment a csv file with the result values
   
5) [Setup a new spike train]
In SingleCellModel_Python\astrocyte\astrocyte.py:
   Line49/50: Specifiy spike train
   Line61: Specify which compartments should be stimulated
   BUT you have to set load_values to false for the first    	time, so that it is not loaded from the spike.ini file
   
6) [Setup the diffusion coefficients, inital ion concentraion]
Diffusion Coefficients can be specified in SingleCellModel_Python\output\test.ini and by setting load_values to true in main.py (line30)
   