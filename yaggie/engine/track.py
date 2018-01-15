from . import linking
import numpy as np

class MinimumTrackpyEngine():
    def __init__(self, max_movement):
        """
        unit of the movement is pixel
        """
        self.max_movement = max_movement

    def __get_trajectory(self, index, labels, maxima):
        """
        trajectory: [(x1, y1, z1) --> t1, (x2, y2, z2) --> t2, ... ]
        """
        trajectory = []
        for i, time in enumerate(labels):
            for j, label in enumerate(time):
                if label == index:
                    trajectory.append(np.array(maxima[i][j]))
        return np.array(trajectory)

    def __get_trajectories(self, labels, maxima):
        result = []
        for i in range(labels.flatten().max() + 1):
            result.append(self.__get_trajectory(i, labels, maxima))
        return np.array(result)

    def run(self, maxima):
        link_result = linking.link_iter(maxima, self.max_movement)
        labels = np.array([np.array(r[1]) for r in link_result])
        trajectories = self.__get_trajectories(labels, maxima)
        self.trajectories = trajectories
        return trajectories
