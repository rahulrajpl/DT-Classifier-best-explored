# Train-Until-Best-Estimate (TUBE) strategy for training best classifier

Trainer program for Decision Tree Classifier.

Program trains a decision tree classifier available from sklearn library.
Training is undertaken using 3 folded RandomizedSearch cross validation of 
random 75% of available dataset. Balance 25% percent is used to test the model.
If the required accuracy is not achieved then program automatically runs again.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Python 3.6 or above installed. Install it as per instructions in official documentation of python
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

### Installing

clone this repo to your local machine when pre requisites are met. That is all.


## Built With

* [sklearn](https://scikit-learn.org/stable/) - Just another data science library
* [python](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Author

* **Rahul Raj** - *Initial work* - [Website](https://randomwalk.in)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Dept. of CSE, IIT Kanpur (C) 2019

