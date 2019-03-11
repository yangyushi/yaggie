import numpy as np
import trackpy as tp


class TrackpyEngine():
    def __init__(self, max_movement, memory=0, max_subnet_size=30):
        """
        unit of the movement is pixel
        """
        self.max_movement = max_movement
        self.memory = memory
        self.max_subnet_size = max_subnet_size

    @staticmethod
    def _check_input(positions, time_points, labels):
        """
        make sure the input is proper
        and sequence in time_points are ordered
        """
        assert len(positions) == len(time_points), "Lengths are not consistent"
        if not isinstance(labels, type(None)):
            assert len(positions) == len(labels), "Lengths are not consistent"
            for p, l in zip(positions, labels):
                assert len(p) == len(l), "Labels and positions are not matched"
        time_points = np.array(time_points)
        order_indice = time_points.argsort()
        ordered_time = time_points[order_indice]
        positions = list(positions)
        positions.sort(key=lambda x: order_indice.tolist())
        return positions, ordered_time, labels

    def __get_trajectory(self, value, link_result, positions, time_points, labels):
        if isinstance(labels, type(None)):
            traj = {'time': [], 'position': []}
        else:
            traj = {'time': [], 'position': [], 'label': []}
        for frame in link_result:
            frame_index, link_labels = frame
            if value in link_labels:
                number_index = link_labels.index(value)
                traj['time'].append(time_points[frame_index])
                traj['position'].append(positions[frame_index][number_index])
                if 'label' in traj:
                    current_label = labels[frame_index][link_labels.index(value)]
                    traj['label'].append(current_label)
        return traj

    def __get_trajectories(self, link_result, positions, time_points, labels):
        trajectories = []
        total_labels = []
        for frame in link_result:
            frame_index, link_labels = frame
            total_labels.append(link_labels)
        for value in set(np.hstack(total_labels)):
            traj = self.__get_trajectory(value, link_result, positions, time_points, labels)
            trajectories.append(traj)
        return trajectories

    def run(self, positions, time_points, labels=None):
        """
        positions: (time, number_of_individuals, dimensions)
        time_points: (time, )
        labels: (time, number_of_individuals)
        * if labels were given, the returned trajecotory will have a 'label' attribute
          which specifies the label values of the individual in different frames
        * time_points may not be continues
        * The unit of time is NOT converted, do the conversion before running

        labels: [(frame_index, [labels, ... ]), ...]
        """
        pos, time, labels = self._check_input(positions, time_points, labels)
        tp.linking.Linker.MAX_SUB_NET_SIZE = self.max_subnet_size
        link_result = tp.link_iter(pos, search_range=self.max_movement, memory=self.memory)
        return self.__get_trajectories(list(link_result), pos, time, labels)
