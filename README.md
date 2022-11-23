# Object tracking travelling using Meta Heuristics

# Description🧠
This project includes different meta-heuristics algorithms:
1. Differential Evolution
2. Gravitational Search Algorithm
3. Harris Hawk Optimization
4. Particle Swarm Optimization
5. Hybrid GSA-PSO

We have used two different objective functions on each of the meta-heuristic algorithms.

### Built With

* [Python](https://www.python.org/)
* [openCV](https://opencv.org/)

# Run Demo💻
You have to download one of the datasets provided by: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

And mention it in `dataset.py` file, so that the algorithm may run on this dataset.

You can follow the steps provided below to start using this project:
1. Clone the repo
   ```sh
   git clone https://github.com/sumit-6/Object-tracking-using-MetaH.git
   ```
2. Install virtualenv first:
   ```
   pip install virtualenv
   ```
3. Create an environment using this command:
   ```
   virtualenv env
   ```
4. Enter into your environment using this command:
   ```
   env\Scripts\activate.bat
   ```
5. Install requirements.txt by running this command:
   ```
   pip install -r requirements.txt
   ```
6. Run the project:
   ```
   python optimizer.py
   ```

