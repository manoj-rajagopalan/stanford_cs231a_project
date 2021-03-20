import numpy as np
import scipy.spatial
import pptk
import random
import sys
from collections import namedtuple

Plane = namedtuple('Plane', ['point', 'normal'])

def RANSAC(pt_cloud, num_iterations, num_trials_per_iteration):
    print(f'--- RANSAC ({len(pt_cloud)} points) ---')
    pt_cloud_indices = np.array(range(len(pt_cloud)))
    plane_ids = np.zeros(len(pt_cloud), dtype=int)
    planes = [Plane(np.zeros(3), np.zeros(3))] # 0-th point is dummy
    plane_id_counter = int(1)
    iter = 0
    for iter in range(num_iterations):
        N = len(pt_cloud_indices)
        print(f'Iter {iter}: {N} points')
        if N < 100:
            break
        residual_pt_cloud = pt_cloud[pt_cloud_indices]
        # kdtree = scipy.spatial.KDTree(residual_pt_cloud)

        max_num_members = 100 # need at least so many points on each plane

        best_plane = Plane(np.zeros(3), np.zeros(3))
        for trial in range(num_trials_per_iteration):
            n0 = np.random.randint(N)
            print(f'\t- trial {iter}.{trial} n0 = {n0}')
            # neighbor_indices = kdtree.query_ball_point(pt_cloud[n0], 0.5)
            # num_neighbors = len(neighbor_indices)
            # print(f'\t- trial {iter}.{trial} found {num_neighbors} neighbors')
            # if num_neighbors < 10:
            #     print(f'\t- trial {iter}.{trial} skipping (< 10)')
            #     continue
            normal = np.zeros(3)
            normal_length = 0.0
            num_tries = 10
            while num_tries > 0 and normal_length < 1.0e-3:
                # n1, n2 = np.random.randint(len(neighbor_indices), size=2)
                n1, n2 = np.random.randint(N, size=2)
                # n1, n2 = neighbor_indices[n1], neighbor_indices[n2]
                n1, n2 = pt_cloud_indices[n1], pt_cloud_indices[n2]
                p0, p1, p2 = pt_cloud[[n0, n1, n2]]
                p1_p0 = p1 - p0
                p2_p0 = p2 - p0
                normal = np.cross(p1_p0, p2_p0)
                normal_length = np.linalg.norm(normal)
                num_tries -= 1
            #/while num_tries

            print(f'\t- trial {iter}.{trial} normal length = {normal_length}')
            if normal_length <= 1.0e-3:
                print(f'\t- trial {iter}.{trial} skipping')
                continue
            #/if
            normal = normal / normal_length
            distances_from_plane = \
                np.abs(np.dot(residual_pt_cloud - p0[np.newaxis, ...], normal))
            plane_membership = distances_from_plane < 0.3
            num_members = np.sum(plane_membership)
            print(f'\t- trial {iter}.{trial} num_members = {num_members} (max = {max_num_members})')
            if num_members > max_num_members:
                best_plane = Plane(p0, normal)
                max_num_members = num_members
            #/if
        #/for trial
            
        if (best_plane.normal != np.zeros(3)).any(): # found valid "best" plane
            plane_ids[pt_cloud_indices[plane_membership]] = plane_id_counter
            plane_id_counter = plane_id_counter + 1
            planes.append(best_plane)
            pt_cloud_indices = pt_cloud_indices[np.logical_not(plane_membership)]
        #/if
    #/for iter

    print(f'{plane_id_counter-1} planes found')
    print(f'After RANSAC, pt_cloud has {len(pt_cloud_indices)} points left')

    return plane_ids, planes
#/RANSAC()

def visualize(pt_cloud, plane_ids):
    pt_cloud = pt_cloud[np.nonzero(plane_ids)]
    plane_ids = plane_ids[np.nonzero(plane_ids)]
    viewer = pptk.viewer(pt_cloud)
    viewer.attributes(plane_ids)
    viewer.set(point_size=0.01)
    # viewer.wait()
    # viewer.close()

def main():
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <.npy file>"
    obj_pt_cloud = np.load(sys.argv[1])
    print("Object point cloud shape =", obj_pt_cloud.shape)

    plane_ids, planes = RANSAC(obj_pt_cloud, 100, 1000)
    visualize(obj_pt_cloud, plane_ids)

    ransaced_pt_cloud = obj_pt_cloud[np.nonzero(plane_ids)]
    plane_ids = plane_ids[np.nonzero(plane_ids)]
    ref_points = np.array([planes[i].point for i in plane_ids])
    normals = np.array([planes[i].normal for i in plane_ids])
    ransaced_pt_cloud_rel = ransaced_pt_cloud - ref_points
    ransaced_pt_cloud_rel = \
        ransaced_pt_cloud_rel - \
        (np.sum(ransaced_pt_cloud_rel * normals, axis=1, keepdims=True) * normals)
    ransaced_pt_cloud_clamped = ransaced_pt_cloud_rel + ref_points
    visualize(ransaced_pt_cloud_clamped, plane_ids)
#/main()

if __name__ == "__main__":
    main()
