
points = [[] for i in range(labels.flatten().max())]
for i, l in enumerate(labels.flatten()):
    x, y, z = np.unravel_index(i, labels.shape)
    points[l-1].append(np.array([x * img.metadata['pixel_size_x'],
                                 y * img.metadata['pixel_size_y'],
                                 z * img.metadata['pixel_size_z']]))

shapes = [[] for i in range(labels.flatten().max())]
for i, group in enumerate(points):
    shapes[i] = spatial.ConvexHull(np.array(group))

for ch in shapes:
    print(ch.vertices)
