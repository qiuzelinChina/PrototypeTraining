# README

This is the code for EEGWaveNet_PrototypeTraining. 



# Run the code

To run the code, you need to do the following:

0. Install the the requirements.

1. Download the dataset into `./Data/Das2016` (Here we use Das-2016 as an example, you can process the other datasets in the same way).

2. Read files `./Data/get_data_for_Das-2016.txt and preprocesse the data accordingly.

3. Change the "base_path" and "DEVICE" in `./utils/cfg.py` according to your environment. "DEVICE" must be 'cpu' or 'cuda'.

4. Run

   * Run `./run/run0.sh` to get the results of CNN.
   * Run `./run/run1.sh` to get the results of DenseNet_3D. Note that "topo" must be True for this model.
   * Run `./run/run2.sh` to get the results of EEGWaveNet. You can change the parameter $K$ in the paper by changing "prototype".

   You can change other parameters according to your needs. 

5. Results

   * You can find the results in `./results`. 



