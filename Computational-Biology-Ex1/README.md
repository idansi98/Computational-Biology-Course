# Computational Biology HW 1
## Setup
No setup is needed, just make sure Ex1.py is in the selected directory.
## Running
Simply run `python3 ./Ex1.py` or `python ./Ex1.py` to run the basic simulation.
The arguments for this program are as follows in this order (you may choose to only use the first X ones)
* P - A decimal number indicating the portion of the grid in which people reside
* L - The cooldown between respeards.
* special/none - a choice between the special configuration of none which is the normal one
* diagonal/none - a choose between if the rumor can spread diagonaly or not
* DrawEveryX - change from 50 to something else if you want to draw every X iterations to speed up the computation, for a closer look we recommend 1.
* MaxRoundsNoImprovement - A parameter which will stop the simulation after X rounds with no improvement
* S1 - the chance for a person of type S1
* S2
* S3
* S4

For example, some runs might look like this:
`python ./Ex1.py 0.9 5 special none 1 50 0.25 0.25 0.25 0.25`
Or
`python ./Ex1.py`
Or
`python3 ./Ex1.py 1.0 3 none`


## Dependencies
We used the following libraries
* Tkinter
* Numpy
* Matplotlib

## Authors

- [@ghsumhubh](https://www.github.com/ghsumhubh)
- [@idansi98](https://github.com/idansi98)

