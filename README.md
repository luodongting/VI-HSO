# VI-HSO: Hybrid Sparse Monocular Visual-Inertial Odometry

This released code is the VIO part of VI-HSO based on HSO.

**Author:** Wenzhe Yang, Yan Zhuang, Dongting Luo , Wei Wang ,and Hong Zhang.

## 1. Related Publications

* Wenzhe Yang, Yan Zhuang, Dongting Luo , Wei Wang ,and Hong Zhang. [VI-HSO: Hybrid Sparse Monocular Visual-Inertial Odometry](https://ieeexplore.ieee.org/document/10218742). In *IEEE Robotics and Automation Letters*.

## 2. Required Dependencies

### boost

c++ Librairies (thread and system are needed). Install with

```
	sudo apt-get install libboost-all-dev
```

### Eigen3

Linear algebra.

```
	sudo apt-get install libeigen3-dev
```

### OpenCV4

Dowload and install instructions can be found at: http://opencv.org. We tested the code on OpenCV 4.2.0.

### Pangolin

Used for 3D visualization & the GUI.
Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

### Sophus, fast and g2o (Included in thirdparty folder)

The three libraries will be compiled automatically using script `build.sh`.

## 2. Build and Compile

We have tested the code on Ubuntu 20.04.

```
	cd VI-HSO
	chmod +x build.sh
	./build.sh
```

This will create the executable `vi_euroc` and `vi_tum` in *bin* folder.

## 3. Run code

### EuRoC Dataset example

```
	cd bin
	./vi_euroc ./Monocular-Inertial/EuRoC.yaml PATH_TO_SEQUENCE_FOLDER ./Monocular-Inertial/EuRoC_TimeStamps/SEQUENCE.txt FILE_NAME
```

### TUM-VI Dataset example

```
	cd bin
	./vi_tum ./Monocular-Inertial/TUM_512.yaml PATH_TO_SEQUENCE_FOLDER ./Monocular-Inertial/TUM_TimeStamps/SEQUENCE.txt FILE_NAME
```

The result of trajectory will be saved in the bin folder.

## 4. License

The source code is released under GPLv3 license. We are still working on improving the code.

If you use VI-HSO in your academic work, please cite:

	@ARTICLE{10218742,
	author={Yang, Wenzhe and Zhuang, Yan and Luo, Dongting and Wang, Wei and Zhang, Hong},
	journal={IEEE Robotics and Automation Letters}, 
	title={VI-HSO: Hybrid Sparse Monocular Visual-Inertial Odometry}, 
	year={2023},
	volume={8},
	number={10},
	pages={6283-6290},
	doi={10.1109/LRA.2023.3305238}}


