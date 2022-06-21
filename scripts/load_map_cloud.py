#!/usr/bin/env python
import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys, os

def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down

def main():
    dataset_dir = sys.argv[1]

    # for j in tqdm(range(2,sub_dataset_num + 1)):
    for subdir in os.listdir(dataset_dir):
        posefile = dataset_dir + subdir + "/optimized_poses.txt"
        transfile = dataset_dir + subdir + "/trans2dataset1.txt"
        scandir = dataset_dir + subdir + "/Scans/"
        df = pd.read_csv(posefile, " ",header=None)

        tr_df = pd.read_csv(transfile, " ", header=[0])
        tr_R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([tr_df["roll"], tr_df["pitch"], tr_df["yaw"]]) )
        T_tr = np.array([[tr_R[0][0], tr_R[0][1], tr_R[0][2], tr_df.iloc[0]["x"]],
                         [tr_R[1][0], tr_R[1][1], tr_R[1][2], tr_df.iloc[0]["y"]],
                         [tr_R[2][0], tr_R[2][1], tr_R[2][2], tr_df.iloc[0]["z"]],
                         [0,0,0,1]])

        df =df.rename(columns={0:"r11",1:"r12",2:"r13",3:"t1",4:"r21",5:"r22",6:"r23",7:"t2",8:"r31",9:"r32",10:"r33",11:"t3"})
        keys = len(df)
        p_loads = np.array([[]])
        for i in tqdm(range(0,keys)):
            kv = df.iloc[i]
            T = np.array([[kv["r11"], kv["r12"], kv["r13"], kv["t1"]],
                          [kv["r21"], kv["r22"], kv["r23"], kv["t2"]],
                          [kv["r31"], kv["r32"], kv["r33"], kv["t3"]],
                          [0,0,0,1]])
            T = np.matmul(T_tr, T)

            pcd_file = scandir + str(i).zfill(6) + ".pcd"
            pcd = o3d.io.read_point_cloud(pcd_file)
            pcd = voxel_down_sample(pcd, 0.4)
            pcd_t = pcd.transform(T)
            p_t_load = np.asarray(pcd_t.points)
            if i == 0:
                p_loads = p_t_load
            else:
                p_loads = np.concatenate((p_loads,p_t_load), axis=0)

        pcd_sum = o3d.geometry.PointCloud()
        pcd_sum.points = o3d.utility.Vector3dVector(p_loads)
        pcd_sum = voxel_down_sample(pcd_sum, 0.4)
        o3d.io.write_point_cloud("../data/" + dataset_dir + ".pcd",pcd_sum)
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit


