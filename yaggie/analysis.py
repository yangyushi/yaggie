import time
import warnings
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

class LabelAnalyser():
    """
    labels --> 3d array 
        shape: (size_x, size_y, size_z)
        value: label value (int)
    """
    def __init__(self, labels, metadata=None):
        self.labels = labels
        self.refined_labels = np.zeros(labels.shape)
        self.metadata = metadata
        if type(self.metadata) == type(None):
            warnings.warn('Analysing without metadata')
        self.max_value = np.max(self.labels)
        self.voxel_numbers = [len(np.where(self.labels == i+1)[0]) for i in range(self.max_value)]

    def label_iter(self):
        for i in range(self.max_value):
            yield self.get_single_label(i+1)

    def get_rg_tensor(self, label):
        """
        radii of gyration tensor
        """
        tensor = np.zeros([3, 3])
        com = self.get_com(label)
        N = label.flatten().sum()
        for i in range(3):
            for j in range(3):
                tensor[i][j] = sum([
                    (xyz[i] - com[i]) * (xyz[j] - com[j]) for xyz in np.array(label.nonzero()).T
                    ]) / N
        return tensor

    def get_single_label(self, label_value):
        single_label = np.zeros(self.labels.shape, dtype=np.uint8)
        single_label[np.where(self.labels == label_value)] = 1
        return single_label

    def get_ch(self, label):
        scatters = np.array(label.nonzero())
        if type(self.metadata) != type(None):
            scatters = scatters * np.array([[self.metadata['pixel_size_x'],
                                             self.metadata['pixel_size_y'],
                                             self.metadata['pixel_size_z']]]).T
        return ConvexHull(scatters.T)

    def get_volume(self, label):
        if type(self.metadata) != type(None):
            voxel_volume = self.metadata['pixel_size_x'] *\
                           self.metadata['pixel_size_y'] *\
                           self.metadata['pixel_size_z']
        else:
            voxel_volume = 1
        return np.count_nonzero(label) * voxel_volume

    @staticmethod
    def get_com(label):
        """
        center_of_mass: (x, y, z)
        """
        scatters = np.array(label.nonzero()).T
        return np.average(scatters, axis=0)

class ZebrabowAnalyser(LabelAnalyser):
    def __init__(self, labels, metadata=None):
        super().__init__(labels, metadata)

    @staticmethod
    def get_single_difference(label, merged_channel, single_channel):
        merged = merged_channel * (label > 0)
        single = single_channel * (label > 0)
        if label.flatten().sum() > 0:
            return (merged - single).flatten().sum() / label.flatten().sum()
        else:
            return False

class TrajectoryAnalyser():
    def __init__(self, time, positions):
        """
        positions: trajectories of different cells, shape (cell_number, time_points, dimenstions)
        The unit of the positions should be µm, which means **VOXEL SIZE** should be considered **BEFORE** construct an analyser!
        This often mean something like this: `trajectory *= voxel`
        times: time points of the trajectory, shape (time_points), the unit should be minutes
        metadata: data contains the voxel information
        """
        assert(type(positions) == np.ndarray)
        assert(type(time) == np.ndarray)
        assert(len(time) == positions.shape[1])
        self.time = time
        self.positions = positions

    def iter_trajectories(self):
        for pos in self.positions:
            yield pos
        
    def get_speed(self):
        distance = self.positions[:, 1:, :] - self.positions[:, :-1, :]
        time_interval = self.time[1:] - self.time[:-1]
        # [t1, t2, ...] --> [[t1, t1, t1], [t2, t2, t2], ...]
        speed = distance / time_interval_xyz
        return speed

    def get_pure_trajectories(self):
        com = np.average(self.positions, axis=0)
        return self.positions - com

    def get_ensemble_msd(self, start=0, stop=None):
        """
        calculate the mean square displacement of cells
        """
        if not stop:
            stop = self.positions.shape[1]
        print(stop)
        msd = np.zeros(stop - start - 1)
        for positions in self.positions:
            msd += self.get_msd(positions, start, stop)
        return msd


    def get_ensemble_acf(self, start=0, stop=None):
        """
        calculate the auto-correlation function of cell velocities
        movements = position_t2 - position_t1 = velocities
        shape of movements: (time_points - 1, dimensions)
        """
        if not stop:
            stop = self.positions.shape[1] - 1
        acf = np.zeros(stop - start)
        for positions in self.positions:
            acf += self.get_acf(positions, start, stop)
        return acf

    def get_ensemble_pdf_dr(self, max_movement=50, bin_number=70, density=False):
        """
        calculate the probability density function of cell displacements
        shape of positions: (time_points, dimensions)
        density: False returns the count, True returns the probablity, see numpy's doc for real understanding
        """
        movements = self.positions[:, 1:, :] - self.positions[:, :-1, :]
        bw = max_movement / bin_number
        bins = np.linspace(0, max_movement + bw/2, bin_number + 1)  # notice the bin_width / 2
        result, bins = np.histogram(abs(movements), bins, density=density)
        return result, bins

    def get_ensemble_pdf_azimuth(self, bin_number = 6, density = False, threshold=0):
        r2d = lambda rad: rad / np.pi * 180
        bw = 180 / bin_number
        bins = np.linspace(0, 180, bin_number + 1)
        angles = []
        for positions in self.positions:
            movements = positions[1:] - positions[:-1]
            for i, move in enumerate(movements):
                a0, e0, r0 = self.cart2sph(move)
                #a0 += np.pi * (a0 < 0)
                a0 = abs(a0)
                if (r0 >= threshold):
                    angles.append(r2d(a0))
        angles = np.array(angles)
        counts, bins = np.histogram(angles, bins, density=density)
        return counts, bins
    
    def get_ensemble_pdf_elevation(self, bin_number=6, density=False, threshold=0):
        r2d = lambda rad: rad / np.pi * 180
        bw = 90 / bin_number
        bins = np.linspace(0, 90, bin_number + 1)
        angles = []
        for positions in self.positions:
            movements = positions[1:] - positions[:-1]
            for i, move in enumerate(movements):
                a0, e0, r0 = self.cart2sph(move)
                #e0 += np.pi/2 * (e0 < 0)
                e0 = abs(e0)
                if (r0 >= threshold):
                    angles.append(r2d(e0))
        angles = np.array(angles)
        counts, bins = np.histogram(angles, bins, density=density)
        return counts, bins

    def get_ensemble_pdf_delta_elevation(self, bin_number=6, density=False, threshold=0):
        r2d = lambda rad: rad / np.pi * 180
        bw = 90 / bin_number
        bins = np.linspace(0, 90, bin_number + 1)
        angles = []
        for positions in self.positions:
            movements = positions[1:] - positions[:-1]
            delta_move = movements[1:] - movements[:-1]
            for i, move in enumerate(delta_move):
                a0, e0, r0 = self.cart2sph(move)
                #e0 += np.pi/2 * (e0 < 0)
                e0 = abs(e0)
                if (r0 >= threshold):
                    angles.append(r2d(e0))
        angles = np.array(angles)
        counts, bins = np.histogram(angles, bins, density=density)
        return counts, bins

    def get_ensemble_pdf_delta_azimuth(self, bin_number=6, density=False, threshold=0):
        r2d = lambda rad: rad / np.pi * 180
        bw = 180 / bin_number
        bins = np.linspace(0, 180, bin_number + 1)
        angles = []
        for positions in self.positions:
            movements = positions[1:] - positions[:-1]
            delta_move = movements[1:] - movements[:-1]
            for i, move in enumerate(delta_move):
                a0, e0, r0 = self.cart2sph(move)
                #a0 += np.pi * (a0 < 0)
                a0 = abs(a0)
                if (r0 >= threshold):
                    angles.append(r2d(a0))
        angles = np.array(angles)
        counts, bins = np.histogram(angles, bins, density=density)
        return counts, bins

    def get_rotation_matrices(self):
        result = []
        for sample in self.positions:
            matrix = self.get_rotation_matrix(sample)
            result.append(matrix)
        return np.array(result)

    @staticmethod
    def get_msd(positions, start=0, stop=None):
        """
        calculate the mean square displacement
        shape of positions: (time_points, dimensions)
        """
        if not stop:
            stop = len(positions)
        msd = np.zeros(stop - start)
        for tau in np.arange(0, len(msd), 1):
            msd[tau] = np.mean(
                           np.sum((positions[tau : stop] - 
                                   positions[    : stop - tau]) ** 2
                                 , axis=-1)  # dx**2 + dy**2 + dz**2, (time_points, 3) ---> (time_points,)
                               , axis=0)  # average along different time points  (time_points) ---> scalar
        return msd[1:]  # trim the data when delta(t) == 0

    @staticmethod
    def get_acf(positions, start=0, stop=None):
        """
        calculate the auto-correlation function of cell velocities
        movements = position_t2 - position_t1 = velocities
        shape of movements: (time_points - 1, dimensions)
        """
        movements = positions[1:] - positions[:-1]
        if not stop:
            stop = len(movements)
        acf = np.zeros(stop - start)
        for tau in np.arange(0, len(acf), 1):
            acf[tau] = np.mean(
                            np.sum((movements[tau : stop] *
                                    movements[    : stop - tau])
                                , axis=-1)  # x + y + z, (time_points, 3) ---> (time_points,)
                            ,axis=0)  # average along different time points  (time_points) ---> scalar
        return abs(acf)
    
    @staticmethod
    def get_pdf_dr(positions, max_movement=50, bin_number=70, density=False):
        """
        calculate the probability density function of cell displacements
        shape of positions: (time_points, dimensions)
        density: False returns the count, True returns the probablity, see numpy's doc for real understanding
        """
        movements = positions[1:] - positions[:-1]
        bin_width = max_movement / bin_number
        bins = np.linspace(0, max_movement, bin_number + 1)
        result, bins = np.histogram(abs(movements), bins, density=density)
        return result, bins
    
    @staticmethod
    def cart2sph(coord):
        """
        Transform Cartesian coordinates to spherical
        coord: (x, y, z)
        """
        coord = np.array(coord)
        azimuth = np.arctan2(coord[1], coord[0])
        elevation = np.arctan2(coord[2], np.sqrt(np.sum(coord[:2]**2, axis=0)))
        radius = np.sqrt(np.sum(coord**2, axis=0))
        if len(coord) == 3:
            return azimuth, elevation, radius
        else:
            raise IndexError("Can't deal with non 3d array")
            
    @staticmethod
    def get_rotation_matrix(positions):
        """
        use svd to find principle axis for the movement
        movement = position_t2 - position_t1
        """
        movements = positions[1:] - positions[:-1]
        u, s, vh = np.linalg.svd(movements)
        v = vh.T
        return v
