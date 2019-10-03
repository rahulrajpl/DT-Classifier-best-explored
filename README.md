# Train-Until-Best-Estimate (TUBE) strategy

Trainer program for training a Decision Tree Classifier with maximum possible accuracy.

Program trains a decision tree classifier available from sklearn library.
Training is undertaken using 3 folded RandomizedSearch cross validation of 
random 75% of available dataset. Balance 25% percent is used to test the model.
If the required accuracy is not achieved then program automatically runs again.

#### "Training phase to find the best model"
![Training in progress _ pic](/imgs/training_bestaccbestFN.png?raw=true "Training phase to find the best model")

#### "Testing phase with accuracy terms"
![test_ pic](/imgs/testingphase.png?raw=true "Test phase to find the best model")


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Python 3.6 or above installed. Install using command
	`sudo apt-get install python3.6`
2. Python3-tk installed. Install using command 
	`sudo apt-get install python3-tk`
3. 'virtualenv' installed. Install using command 
	`sudo apt-get install virtualenv`
4. 'pip' installed. Install using command 
	`sudo apt-get install -y python3-pip`

5. It is recommended that a virtual environment be created before testing this program Same can be created and activated using following command: `virtualenv venv` and `source venv/bin/activate`

6. All dependencies of this program is mentioned in the requirements.txt file.
Ensure that they are installed using the command `pip install -r requirements.txt`
	
Note: This program is developed and tested only on ubuntu 18.04. 

### Installing and Deployment

clone this repo to your local machine when pre requisites are met. That is all. Walk through following 
steps to reproduce the results claimed.

1. Open Terminal in that folder
2. create a virtual environment using command `virtualenv venv`
3. Activate Virtual Env using command `source venv/bin/activate`
4. Install all the dependencies using command `pip install -r requirements.txt`
5. Once the installation is completed, run command `python3 train.py`
6. Once training is completed, run command `python3 test.py n_samples n_iterations`

Example usage
```
python3 test.py 2000 10
python3 test.py 100 10
python3 test.py 4150 10
```

## Built With

* [sklearn](https://scikit-learn.org/stable/) - Just another data science library :)

## Contributing

Drop a pull request with details.

## Authors

* **Rahul Raj** - *Initial work* - [Website](https://randomwalk.in)
* **Sai Charan** - *Contributor* -[Website](http://pvsaicharan.in/)
* **K Parvez** - *Contributor*

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Dept. of CSE, IIT Kanpur (C) 2019

