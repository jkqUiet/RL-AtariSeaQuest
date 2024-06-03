import asyncio
import subprocess
import concurrent.futures
from time import sleep


def run_command(command):
    print(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10  # Timeout set to 10 seconds
        )
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        stdout = "Timeout expired"
        stderr = ""
    except Exception as e:
        stdout = ""
        stderr = str(e)
    return stdout, stderr
def main():
    tasksseq = [
        "python QLearning.py 1 relu QLearn1 QLearn1",
        "python QLearning.py 2 relu QLearn2 QLearn2",
        "python QLearning.py 3 relu QLearn3 QLearn3",
        "python QLearning.py 1 sigmoid QLearn12 QLearn12",
        "python QLearning.py 2 sigmoid QLearn22 QLearn22",
        "python QLearning.py 3 sigmoid QLearn32 QLearn32",
        "python DoubleQLearning.py 1 relu DoubleQLearn1 DoubleQLearn1",
        "python DoubleQLearning.py 2 relu DoubleQLearn2 DoubleQLearn2",
        "python DoubleQLearning.py 3 relu DoubleQLearn3 DoubleQLearn3",
        "python DoubleQLearning.py 1 sigmoid DoubleQLearn12 DoubleQLearn12",
        "python DoubleQLearning.py 2 sigmoid DoubleQLearn22 DoubleQLearn22",
        "python DoubleQLearning.py 3 sigmoid DoubleQLearn32 DoubleQLearn32",
    ]

    #argumenty programu : pocet vrstiev [1..3], aktivacne funkcie, csv output nazov, txt output nazov,
    tasks = [
        "python QLearning.py 1 relu QLearn1 QLearn1",
        "python QLearning.py 2 relu QLearn2 QLearn2",
        "python QLearning.py 3 relu QLearn3 QLearn3",
    ]
    tasks1 = {
        "python QLearning.py 1 sigmoid QLearn12 QLearn12",
        "python QLearning.py 2 sigmoid QLearn22 QLearn23",
        "python QLearning.py 3 sigmoid QLearn32 QLearn33",
    }
    tasks2 = {
        "python DoubleQLearning.py 1 relu DoubleQLearn1 DoubleQLearn1",
        "python DoubleQLearning.py 2 relu DoubleQLearn2 DoubleQLearn2",
        "python DoubleQLearning.py 3 relu DoubleQLearn3 DoubleQLearn3",
    }
    tasks3 = {
        "python DoubleQLearning.py 1 sigmoid SARSALearning12 SARSALearning12",
        "python DoubleQLearning.py 2 sigmoid SARSALearning22 SARSALearning22",
        "python DoubleQLearning.py 3 sigmoid SARSALearning32 SARSALearning32",
    }
    tasks4 = {
        "python SARSALearning.py 1 relu SARSALearning12 SARSALearning12",
        "python SARSALearning.py 2 relu SARSALearning22 SARSALearning22",
        "python SARSALearning.py 3 relu SARSALearning32 SARSALearning32",
    }
    tasks5 = {
        "python SARSALearning.py 1 sigmoid SARSALearning12 SARSALearning12",
        "python SARSALearning.py 2 sigmoid SARSALearning22 SARSALearning22",
        "python SARSALearning.py 3 sigmoid SARSALearning32 SARSALearning32",
    }
    results = []

    """with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, task) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    sleep(5)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, task) for task in tasks1]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    sleep(5)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, task) for task in tasks2]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    sleep(5)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, task) for task in tasks3]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]"""
    #sleep(5)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_command, task) for task in tasks4]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    #sleep(5)
    #with concurrent.futures.ThreadPoolExecutor() as executor:
        #futures = [executor.submit(run_command, task) for task in tasks5]
        #results = [future.result() for future in concurrent.futures.as_completed(futures)]
    #sleep(5)
    #for task in tasksseq:
       # run_command(task)
        #sleep(5)

if __name__ == "__main__":
    main()