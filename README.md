# mdp_network

Implementation of MDP networks for modeling and policy generation used to synthesize policies for the two papers:

 - Petter Nilsson, Thomas Gurriet, Andrew Singletary and Aaron D. Ames, **Demonstrating Cooperative Multi-Robot Task Planning in Partially Known Environments**

 - Petter Nilsson, Sofie Haesaert, Rohan Thakker, Kyohei Otsu, Cristian-Ioan Vasile, Ali-akbar Agha-Mohammadi, Aaron D. Ames and Richard M. Murray, **Toward Specification-Guided Active Mars Exploration for Cooperative Robot Teams**, in *Proceedings of Robotics: Science and Systems Conference*, 2018

## Installation

This package is developed for Python 3.6. Install necessary packages

    sudo apt install graphviz-dev 
    pip install -r requirements.txt 
    
Install the ``best`` package as follows:

    python setup.py install

For development (create links to source code, no need to reinstall after making changes):

    python setup.py develop

Run tests:

    nosetests
    
To use ROS-based controllers, in addition these packages are needed

    pip install rospkg catkin-tools pyyaml


# TODO

 - Base class for policies, solvers return subclasses
 - Competing types of value iteration: 
    * [x] sparse tables (for MDPs)
    * [ ] mtBDD? (for MDPs)
    * [ ] VDC
    * [ ] DQN via openai.baseline
    * [ ] pbVI by converting Rohan's code?


