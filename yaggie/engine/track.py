import numpy as np
import trackpy as tp


class TrackpyEngine():
    def __init__(self, max_movement, memory=0):
        """
        unit of the movement is pixel
        """
        self.max_movement = max_movement
        self.memory = memory
        self.labels = None

    @staticmethod
    def _check_input(positions, time_points):
        """
        make sure the input is proper
        and sequence in time_points are ordered
        """
        assert len(positions) == len(time_points), "Lengths are not consistent"
        time_points = np.array(time_points)
        order_indice = time_points.argsort()
        ordered_time = time_points[order_indice]
        positions.sort(key=lambda x: order_indice.tolist())
        return positions, ordered_time

    def __get_trajectory(self, value, link_result, positions, time_points):
        traj = {'time': [], 'position': []}
        for frame in link_result:
            frame_index, labels = frame
            if value in labels:
                number_index = labels.index(value)
                traj['time'].append(time_points[frame_index])
                traj['position'].append(positions[frame_index][number_index])
        return traj

    def __get_trajectories(self, link_result, positions, time_points):
        trajectories = []
        total_labels = []
        for frame in link_result:
            frame_index, labels = frame
            total_labels.append(labels)
        for value in set(np.hstack(total_labels)):
            traj = self.__get_trajectory(value, link_result, positions, time_points)
            trajectories.append(traj)
        return trajectories

    def run(self, positions, time_points):
        """
        positions: (time, number_of_individuals, dimensions)
        time_points: (time, )
        * time_points may not be continues
        * The unit of time is NOT converted, do the conversion before running

        labels: [(frame_index, [labels, ... ]), ...]
        """
        pos, time = self._check_input(positions, time_points)
        link_result = tp.link_iter(pos, search_range=self.max_movement, memory=self.memory)
        return self.__get_trajectories(list(link_result), pos, time)
