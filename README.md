# LIO_Connector
tool for generate and visualize inter-dataset loop closure for long-term SLAM
```

├──LIO_connector
   ├── ...
   ├── src                    
   │   ├── ...           
   │   └── mian.cpp                # Tool for gathering the inter dataset loop closure based on DiSco SLAM
   ├── scripts
   │   └── load_map_cloud.py       # Tested with Python 2.7
   └── launch
       └── visualization.launch
```

## How to use? 
- run load_map_cloud.py for the pcd pointcloud of the whole map
```
python2 load_map_cloud.py <directory to the datasets, ie:NCLT-1-3>
```
- change the file name in visualization.launch to the name of the output file from load_map_cloud.py
- run visualization.launch to see the total point cloud (it takes some time to load the point cloud.)
```
roslaunch lio_connector visualization.launch
```
