# Train-Until-Best-Estimate (TUBE) strategy for training best classifier

Trainer program for Decision Tree Classifier.

Program trains a decision tree classifier available from sklearn library.
Training is undertaken using 3 folded RandomizedSearch cross validation of 
random 75% of available dataset. Balance 25% percent is used to test the model.
If the required accuracy is not achieved then program automatically runs again.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. Python 3.6 or above installed. 
		Install it as per instructions in official documentation of python
	2. Python3-tk installed. 
		Install using command 
		>> sudo apt-get install python3-tk
	3. 'virtualenv' installed. 
		Install using command 
		>> sudo apt-get install virtualenv
	4. 'pip' installed. 
		Install using command 
		>> sudo apt-get install -y python3-pip

	5. It is recommended that a virtual environment be created before testing this program
	Same can be created and activated using following command:

		>> virtualenv venv
		>> source venv/bin/activate

	6. All dependencies of this program is mentioned in the requirements.txt file.
	Ensure that they are installed using the command given below:

		>> pip install -r requirements.txt
	
	Note: This program is developed and tested only on ubuntu 18.04. 

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

