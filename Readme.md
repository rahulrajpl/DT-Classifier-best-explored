
# Instructions

Following are the two methods to reproduce the results claimed in the report. Preferred method is Method 1.

## Method 1 (Easy way)

### Pre-requisites on host machine
-------------

Linux OS (preferably an Ubuntu 18.04) installed with docker container. Same can be installed using instruction at step 1 [here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04). Once the installation is finished, run following command to download the docker image.
```
docker pull rahulrajpl/clf
```
This docker image already has the source files. Once the image is downloaded, run the test commands shown in the examples section directly to test the model and evaluate.
	

## Method 2

### Pre-requisites on host machine
-------------

Ensure that **Python3.6**, **python3-tk**, **virtualenv** and **pip**. If not available, install it using following commands
```
>> sudo apt-get install python3.6
>> sudo apt-get install python3-tk
>> sudo apt-get install virtualenv
>> sudo apt-get install -y python3-pip
```

### Setting up the source directory
-------------

1. Extract the zip file submitted
2. Go to the source folder and open a terminal window
3. Activate a virtual environment. This step is optional, however recommended.
   ```
   >> virtualenv venv
   >> source venv/bin/activate
   ```
4. Install all the dependencies using command 
   ```
   >> pip install -r requirements.txt
   ```
5. Once the installation is completed, run following command to initialize the training phase. This is an optional step, as a trained model, named *'best.model'*, is already in the source folder.
	```
	>> python3 train.py
	```
6.  For testing run following command 
	
	```
	>> python3 test.py <n_samples> <n_iterations> [OPTIONAL model_file]```

###	Example usage
------------

If training is carried out freshly, then testing can be done as shown below
```
>> python3 test.py 2000 10
```

```
>> python3 test.py 2000 10 best.model 
```
Above command will test the model for 10 different samples of size 2000 rows. Other examples are shown below
```
>> python3 test.py 1000 20 best.model
>> python3 test.py 100 10 best.model
>> python3 test.py 1500 20 best.model
```

### Note: 
Documentation of source code for training program can be viewed using following command from the terminal

```
>> pydoc train
```
