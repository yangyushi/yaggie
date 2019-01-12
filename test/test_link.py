import sys
sys.path.append('../yaggie/engine')
from track import TrackpyEngine
import numpy as np


def get_data(particle_number, frames, speed, missing_ratio=0.1):
    minimum_tracked = int(np.floor(particle_number * (1 - missing_ratio)))
    tracked_numbers = np.random.randint(minimum_tracked, particle_number, 50)
    time_points = np.random.permutation(frames) 

    np.random.seed(0)
    x0 = np.random.uniform(-100, 100, (particle_number, 3))
    full_positions = [x0]  # t0
    real_positions = [x0]  # some particles missing in real tracking result
    random_indices = [list(range(particle_number))]

    for f in range(frames - 1): 
        prev = full_positions[-1]
        current = prev + np.random.normal(0, speed, prev.shape)
        random_order = np.random.permutation(particle_number)[:tracked_numbers[f]]
        full_positions.append(current)
        real_positions.append(current[random_order])
        random_indices.append(random_order)

    return time_points, full_positions, real_positions, random_indices

if __name__ == "__main__":
    particle_num = 10
    frame_num = 20
    move_speed = 2
    times, full_coms, coms, random_indices = get_data(particle_num, frame_num, move_speed)
    te = TrackpyEngine(20, 5)
    trajs = te.run(coms, times, random_indices)
    assert len(trajs) == particle_num, f'for {particle_num} particels, {len(trajs)} trajectories were found'
    for i, traj in enumerate(trajs):
        label_set = set(traj['label'])
        assert len(label_set) == 1, f'for trajectory {i}, label set is {label_set}'
    print("Linking test passed!")
