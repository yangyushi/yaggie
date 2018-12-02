import numpy as np
import trackpy as tp


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
        link_result = tp.link_iter(maxima, self.max_movement)
        labels = np.array([np.array(r[1]) for r in link_result])
        trajectories = self.__get_trajectories(labels, maxima)
        self.trajectories = trajectories
        return trajectories


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
        make sure the shape is right
        and sequence in time_points are ordered
        """
        positions = np.array(positions)
        time_points = np.array(time_points)
        assert positions.shape[0] == time_points.shape[0], "Lengths are not consistent"
        order_indice = np.argsort(time_points, axis=0)
        ordered_pos = positions[order_indice]
        ordered_time = time_points[order_indice]
        return ordered_pos, ordered_time

    def __get_trajectory(self, value, link_result, positions, time_points):
        traj = {'time': [], 'position': []}
        for frame in link_result:
            frame_index, labels = frame
            if value in labels:
                number_index = labels.index(value)
                traj['time'].append(time_points[frame_index])
                traj['position'].append(positions[frame_index, number_index])
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


if __name__ == "__main__":
    # linking random numbers would always result in broken trajectories
    import sys
    sys.path.append('..')
    from utility import plot_trajectories as plot
    from utility import plot_trajectory as plot_one
    te = TrackpyEngine(150)
    frames = 10
    number = 5
    positions = np.random.random((frames, number, 3)) * 100
    time_points = np.random.permutation(frames)
    positions += np.expand_dims(np.expand_dims(time_points, -1), -1) * 50
    trajs = te.run(positions, time_points)
    plot(trajs, projection=False)
    plot_one(trajs[2])
    print('trajectory #0 looks like this: ')
    print(trajs[0])
