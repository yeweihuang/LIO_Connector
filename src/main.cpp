//
// Created by yewei on 6/16/22.
//
#include <ros/ros.h>

//pcl
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include <Eigen/Dense>

#include "nabo/nabo.h"
#include "scanContext/scanContext.h"
#include "fast_max-clique_finder/src/findClique.h"

//expression graph
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/slam/dataset.h>

//gtsam
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

using namespace  std;
using namespace gtsam;

typedef pcl::PointXYZI PointType;

class LioConnector {

private:

//SC

    ScanContext *_scan_context_factory;
    const int _knn_feature_dim = 32;
    const int _num_sectors = 60;
    const int _max_range = 30;
    const int _num_nearest_matches = 50;

    const int _num_match_candidates = 1;
    const double _loop_thres = 0.2;
    const double _icp_thres = 3;

    const string _pcm_matrix_folder = "/home/yewei/ros/lio_sam/src/lio_connector/data/";
    const int _pcm_start_threshold = 5;
    const double _pcm_thres = 10;

    string _dataset_dir;
    int _sub_dataset_num;

    string _robot_this;
    int _robot_this_th;

    int _num_bin;
    unordered_map<int, ScanContextBin> _bin_with_id;
    Nabo::NNSearchF *_nns = NULL; //KDtree
    Eigen::MatrixXf _target_matrix;

    std::vector<std::pair<int, int>> _idx_nearest_list;
    std::vector<int> _robot_received_list;
    std::unordered_map<int, std::vector<PointTypePose> > _global_map_trans;
    std::unordered_map<int, PointTypePose> _global_map_trans_optimized;
    std::unordered_map< int, std::vector<int> > _loop_accept_queue;

//for pcm & graph
//first: source pose; second: source pose in target; third: icp fitness score;
    std::unordered_map<int, std::vector<std::tuple<gtsam::Pose3, gtsam::Pose3, float> > > _pose_queue;
//first: target, second: source, third: relative transform;
    std::unordered_map<int, std::vector<std::tuple<int, int, gtsam::Pose3> > > _loop_queue;

public:

    LioConnector(string dataset_dir, int sub_dataset_num){
        _dataset_dir = dataset_dir;
        _sub_dataset_num = sub_dataset_num;

        _scan_context_factory =
                new ScanContext(_max_range, _knn_feature_dim, _num_sectors);
        _num_bin = 0;
    }

    void buildKDTree(ScanContextBin bin) {
        _num_bin++;
        //store data received
        _bin_with_id.emplace(std::make_pair(_num_bin - 1, bin));


        PointType tmp_pose;
        tmp_pose.x = bin.pose.x;
        tmp_pose.y = bin.pose.y;
        tmp_pose.z = bin.pose.z;
        tmp_pose.intensity = _num_bin - 1;

        //add the latest ringkey
        _target_matrix.conservativeResize(_knn_feature_dim, _num_bin);
        _target_matrix.block(0, _num_bin - 1, _knn_feature_dim, 1) =
                bin.ringkey.block(0, 0, _knn_feature_dim, 1);
        //add the target matrix to nns
        _nns = Nabo::NNSearchF::createKDTreeLinearHeap(_target_matrix);
    }

    float distBtnScanContexts(Eigen::MatrixXf bin1, Eigen::MatrixXf bin2, int &idx) {
        Eigen::VectorXf sim_for_each_cols(_num_sectors);
        for (int i = 0; i < _num_sectors; i++) {

            //shift
            int one_step = 1;
            Eigen::MatrixXf bint = circShift(bin1, one_step);
            bin1 = bint;

            //compare
            float sum_of_cos_sim = 0;
            int num_col_engaged = 0;

            for (int j = 0; j < _num_sectors; j++) {
                Eigen::VectorXf col_j_1(_knn_feature_dim);
                Eigen::VectorXf col_j_2(_knn_feature_dim);
                col_j_1.block(0, 0, _knn_feature_dim, 1) = bin1.block(0, j, _knn_feature_dim, 1);
                col_j_2.block(0, 0, _knn_feature_dim, 1) = bin2.block(0, j, _knn_feature_dim, 1);

                if (col_j_1.isZero() || col_j_2.isZero())
                    continue;

                //calc sim
                float cos_similarity = col_j_1.dot(col_j_2) / col_j_1.norm() / col_j_2.norm();
                sum_of_cos_sim += cos_similarity;

                num_col_engaged++;
            }
            //devided by num_col_engaged: So, even if there are many columns that are excluded from the calculation, we can get high scores if other columns are well fit.
            sim_for_each_cols(i) = sum_of_cos_sim / float(num_col_engaged);

        }
        Eigen::VectorXf::Index idx_max;
        float sim = sim_for_each_cols.maxCoeff(&idx_max);
        idx = idx_max;
        //get the corresponding angle of the maxcoeff
        float dist = 1 - sim;
        return dist;

    }

    void KNNSearch(ScanContextBin bin, int robot_id) {
        if (_num_nearest_matches >= _num_bin) {
            return;//if not enough candidates, return
        }

        int num_neighbors = _num_nearest_matches;

        //search n nearest neighbors
        Eigen::VectorXi indices(num_neighbors);
        Eigen::VectorXf dists2(num_neighbors);

        _nns->knn(bin.ringkey, indices, dists2, num_neighbors);

        int idx_candidate, rot_idx;
        float distance_to_query;

        //first: dist, second: idx in bin, third: rot_idx
        std::vector<std::tuple<float, int, int>> idx_list;
        for (int i = 0; i < std::min(num_neighbors, int(indices.size())); ++i) {
            //check if the searching work normally
            if (indices.sum() == 0)
                continue;
            idx_candidate = indices(i);

            if (idx_candidate >= _num_bin || idx_candidate < 0){
                continue;
            }
            // if the candidate & source belong to same robot, skip
            if (bin.robotname == _bin_with_id.at(idx_candidate).robotname){
                continue;
            }
            //compute the dist with full scancontext info
            distance_to_query = distBtnScanContexts(bin.bin, _bin_with_id.at(idx_candidate).bin, rot_idx);

            if (distance_to_query > _loop_thres)
                continue;

            //add to idx list
            idx_list.emplace_back(std::make_tuple(distance_to_query, idx_candidate, rot_idx));
        }

        _idx_nearest_list.clear();

        if (idx_list.size() == 0)
            return;

        //find nearest scan contexts
        std::sort(idx_list.begin(), idx_list.end());
        for (int i = 0; i < std::min(_num_match_candidates, int(idx_list.size())); i++) {
            std::tie(distance_to_query, idx_candidate, rot_idx) = idx_list[i];
            _idx_nearest_list.emplace_back(std::make_pair(idx_candidate, rot_idx));
        }
        idx_list.clear();
    }

    int robotID2Number(std::string robo) {
        return robo.back() - '0';
    }

    pcl::PointCloud<PointType>::Ptr
    transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z,
                                                          transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < cloudSize; ++i) {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x =
                    transCur(0, 0) * pointFrom->x + transCur(0, 1) * pointFrom->y + transCur(0, 2) * pointFrom->z +
                    transCur(0, 3);
            cloudOut->points[i].y =
                    transCur(1, 0) * pointFrom->x + transCur(1, 1) * pointFrom->y + transCur(1, 2) * pointFrom->z +
                    transCur(1, 3);
            cloudOut->points[i].z =
                    transCur(2, 0) * pointFrom->x + transCur(2, 1) * pointFrom->y + transCur(2, 2) * pointFrom->z +
                    transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    PointTypePose icpRelativeMotion(pcl::PointCloud<PointType>::Ptr source,
                                    pcl::PointCloud<PointType>::Ptr target,
                                    PointTypePose pose_source) {
        pcl::VoxelGrid<PointType> _downsize_filter_icp;
        _downsize_filter_icp.setLeafSize(0.4, 0.4, 0.4);
        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        _downsize_filter_icp.setInputCloud(source);
        _downsize_filter_icp.filter(*cloud_temp);
        *source = *cloud_temp;

        _downsize_filter_icp.setInputCloud(target);
        _downsize_filter_icp.filter(*cloud_temp);
        *target = *cloud_temp;

        //Align clouds
        icp.setInputSource(source);
        icp.setInputTarget(target);
        pcl::PointCloud<PointType>::Ptr unused_result(
                new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        PointTypePose pose_from;

        if (icp.hasConverged() == false) {
            pose_from.intensity = -1;
            return pose_from;
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;

        correctionLidarFrame = icp.getFinalTransformation();  // get transformation in camera frame

        pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);

//        if(std::min(_robot_id_th, _robot_this_th) == 1 && std::max(_robot_id_th, _robot_this_th) == 2){
//            publishCloud(&_pub_target_cloud, target, _cloud_header.stamp, "/jackal1/odom");
//
//            PointTypePose ptp;
//            ptp.x = x; ptp.y = y; ptp.z = z; ptp.roll = roll; ptp.yaw = yaw; ptp.pitch = pitch;
//            publishCloud(&_pub_match_cloud, transformPointCloud(source, &ptp), _cloud_header.stamp, "/jackal1/odom");
//        }


        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pcl::getTransformation(pose_source.x, pose_source.y, pose_source.z,
                                                        pose_source.roll, pose_source.pitch, pose_source.yaw);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;

        // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);

        pose_from.x = x;
        pose_from.y = y;
        pose_from.z = z;

        pose_from.yaw = yaw;
        pose_from.roll = roll;
        pose_from.pitch = pitch;

        pose_from.intensity = icp.getFitnessScore();

        return pose_from;

    }

    bool getInitialGuess(ScanContextBin bin, int idx_nearest, int min_idx, int _robot_id_th) {

        int id0 = idx_nearest, id1 = _num_bin - 1;

        ScanContextBin bin_nearest;
        PointTypePose source_pose_initial, target_pose;

        float sc_pitch = (min_idx + 1) * M_PI * 2 / _num_sectors;
        if (sc_pitch > M_PI)
            sc_pitch -= (M_PI * 2);

        int robot_id_this = robotID2Number(bin.robotname);

        auto robot_id_this_ite = std::find(_robot_received_list.begin(), _robot_received_list.end(), robot_id_this);

        //record all received robot id from other robots
        if (robot_id_this_ite == _robot_received_list.end() && robot_id_this != _robot_id_th)
            _robot_received_list.push_back(robot_id_this);

        //  exchange if source has a prior robot id (the last character of the robot name is smaller) (first > second)
        if (robot_id_this < _robot_id_th) {
            bin_nearest = bin;
            bin = _bin_with_id.at(idx_nearest);

            id0 = _num_bin - 1;
            id1 = idx_nearest;

            sc_pitch = -sc_pitch;
        } else
            bin_nearest = _bin_with_id.at(idx_nearest);

        _robot_this = bin_nearest.robotname;
        _robot_this_th = robotID2Number(_robot_this);

        //get initial guess from scancontext
        target_pose = bin_nearest.pose;

        //find the pose constrain
        if (_global_map_trans_optimized.find(_robot_this_th) != _global_map_trans_optimized.end()) {
            PointTypePose trans_to_that = _global_map_trans_optimized[_robot_this_th];
            Eigen::Affine3f t_source2target = pcl::getTransformation(trans_to_that.x, trans_to_that.y, trans_to_that.z,
                                                                     trans_to_that.roll, trans_to_that.pitch,
                                                                     trans_to_that.yaw);
            Eigen::Affine3f t_source = pcl::getTransformation(bin.pose.x, bin.pose.y, bin.pose.z, bin.pose.roll,
                                                              bin.pose.pitch, bin.pose.yaw);
            Eigen::Affine3f t_initial_source = t_source2target * t_source;
            pcl::getTranslationAndEulerAngles(t_initial_source, source_pose_initial.x, source_pose_initial.y,
                                              source_pose_initial.z,
                                              source_pose_initial.roll, source_pose_initial.pitch,
                                              source_pose_initial.yaw);
            //if too far away, return false

        } else if (abs(sc_pitch) < 0.3) {
            source_pose_initial = target_pose;
        } else {
            Eigen::Affine3f sc_initial = pcl::getTransformation(0, 0, 0,
                                                                0, 0, sc_pitch);
            Eigen::Affine3f t_target = pcl::getTransformation(target_pose.x, target_pose.y, target_pose.z,
                                                              target_pose.roll, target_pose.pitch, target_pose.yaw);
            Eigen::Affine3f t_initial_source = sc_initial * t_target;
            // pre-multiplying -> successive rotation about a fixed frame

            pcl::getTranslationAndEulerAngles(t_initial_source, source_pose_initial.x, source_pose_initial.y,
                                              source_pose_initial.z,
                                              source_pose_initial.roll, source_pose_initial.pitch,
                                              source_pose_initial.yaw);
            source_pose_initial.x = target_pose.x;
            source_pose_initial.y = target_pose.y;
            source_pose_initial.z = target_pose.z;
        }

        PointTypePose pose_source_lidar = icpRelativeMotion(transformPointCloud(bin.cloud, &source_pose_initial),
                                                            transformPointCloud(bin_nearest.cloud, &target_pose),
                                                            source_pose_initial);

        if (pose_source_lidar.intensity == -1 || pose_source_lidar.intensity > _icp_thres)
            return false;

        //1: jackal0, 2: jackal1
        gtsam::Pose3 pose_from =
                gtsam::Pose3(gtsam::Rot3::RzRyRx(bin.pose.roll, bin.pose.pitch, bin.pose.yaw),
                             gtsam::Point3(bin.pose.x, bin.pose.y, bin.pose.z));

        gtsam::Pose3 pose_to =
                gtsam::Pose3(
                        gtsam::Rot3::RzRyRx(pose_source_lidar.roll, pose_source_lidar.pitch, pose_source_lidar.yaw),
                        gtsam::Point3(pose_source_lidar.x, pose_source_lidar.y, pose_source_lidar.z));

        gtsam::Pose3 pose_target =
                gtsam::Pose3(gtsam::Rot3::RzRyRx(target_pose.roll, target_pose.pitch, target_pose.yaw),
                             gtsam::Point3(target_pose.x, target_pose.y, target_pose.z));

        auto ite = _pose_queue.find(_robot_this_th);
        if (ite == _pose_queue.end()) {
            std::vector<std::tuple<gtsam::Pose3, gtsam::Pose3, float> > new_pose_queue;
            std::vector<std::tuple<int, int, gtsam::Pose3> > new_loop_queue;
            _pose_queue.emplace(std::make_pair(_robot_this_th, new_pose_queue));
            _loop_queue.emplace(std::make_pair(_robot_this_th, new_loop_queue));
        }


        _pose_queue[_robot_this_th].emplace_back(std::make_tuple(pose_from, pose_to, pose_source_lidar.intensity));
        _loop_queue[_robot_this_th].emplace_back(std::make_tuple(id0, id1, pose_to.between(pose_target)));

        return true;
    }

    bool getInitialGuesses(ScanContextBin bin, int key) {
        if (_idx_nearest_list.size() == 0) {
            return false;
        }
        bool new_candidate_signal = false;
        for (auto it: _idx_nearest_list) {
            new_candidate_signal = getInitialGuess(bin, it.first, it.second, key);
        }
        return new_candidate_signal;
    }

    float residualPCM(gtsam::Pose3 inter_jk, gtsam::Pose3 inter_il, gtsam::Pose3 inner_ij, gtsam::Pose3 inner_kl,
                      float intensity) {
        gtsam::Pose3 inter_il_inv = inter_il.inverse();
        gtsam::Pose3 res_pose = inner_ij * inter_jk * inner_kl * inter_il_inv;
        gtsam::Vector6 res_vec = gtsam::Pose3::Logmap(res_pose);

        Eigen::Matrix<double, 6, 1> v;
        v << intensity, intensity, intensity,
                intensity, intensity, intensity;
        Eigen::Matrix<double, 6, 6> m_cov = v.array().matrix().asDiagonal();

        return sqrt(res_vec.transpose() * m_cov * res_vec);
    }

    Eigen::MatrixXi computePCMMatrix(std::vector<std::tuple<int, int, gtsam::Pose3> > loop_queue_this) {
        Eigen::MatrixXi PCMMat;
        PCMMat.setZero(loop_queue_this.size(), loop_queue_this.size());
        int id_0, id_1;
        gtsam::Pose3 z_aj_bk, z_ai_bl;
        gtsam::Pose3 z_ai_aj, z_bk_bl;
        gtsam::Pose3 t_ai, t_aj, t_bk, t_bl;

        for (unsigned int i = 0; i < loop_queue_this.size(); i++) {
            std::tie(id_0, id_1, z_aj_bk) = loop_queue_this[i];
            PointTypePose tmp_pose_0 = _bin_with_id.at(id_0).pose;
            PointTypePose tmp_pose_1 = _bin_with_id.at(id_1).pose;
            t_aj = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_pose_0.roll, tmp_pose_0.pitch, tmp_pose_0.yaw),
                                gtsam::Point3(tmp_pose_0.x, tmp_pose_0.y, tmp_pose_0.z));
            t_bk = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_pose_1.roll, tmp_pose_1.pitch, tmp_pose_1.yaw),
                                gtsam::Point3(tmp_pose_1.x, tmp_pose_1.y, tmp_pose_1.z));

            for (unsigned int j = i + 1; j < loop_queue_this.size(); j++) {
                std::tie(id_0, id_1, z_ai_bl) = loop_queue_this[j];
                PointTypePose tmp_pose_0 = _bin_with_id.at(id_0).pose;
                PointTypePose tmp_pose_1 = _bin_with_id.at(id_1).pose;
                t_ai = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_pose_0.roll, tmp_pose_0.pitch, tmp_pose_0.yaw),
                                    gtsam::Point3(tmp_pose_0.x, tmp_pose_0.y, tmp_pose_0.z));
                t_bl = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_pose_1.roll, tmp_pose_1.pitch, tmp_pose_1.yaw),
                                    gtsam::Point3(tmp_pose_1.x, tmp_pose_1.y, tmp_pose_1.z));
                z_ai_aj = t_ai.between(t_aj);
                z_bk_bl = t_bk.between(t_bl);
                float resi = residualPCM(z_aj_bk, z_ai_bl, z_ai_aj, z_bk_bl, 1);
                if (resi < _pcm_thres)
                    PCMMat(i, j) = 1;
                else
                    PCMMat(i, j) = 0;
            }
        }
        return PCMMat;
    }

    void printPCMGraph(Eigen::MatrixXi pcm_matrix, std::string file_name) {
        // Intialization
        int nb_consistent_measurements = 0;

        // Format edges.
        std::stringstream ss;
        for (int i = 0; i < pcm_matrix.rows(); i++) {
            for (int j = i; j < pcm_matrix.cols(); j++) {
                if (pcm_matrix(i, j) == 1) {
                    ss << i + 1 << " " << j + 1 << std::endl;
                    nb_consistent_measurements++;
                }
            }
        }

        // Write to file
        std::ofstream output_file;
        output_file.open(file_name);
        output_file << "%%MatrixMarket matrix coordinate pattern symmetric" << std::endl;
        output_file << pcm_matrix.rows() << " " << pcm_matrix.cols() << " " << nb_consistent_measurements << std::endl;
        output_file << ss.str();
        output_file.close();
    }


    bool incrementalPCM() {
        if (_pose_queue[_robot_this_th].size() < _pcm_start_threshold)
            return false;

        //perform pcm for all robot matches

        Eigen::MatrixXi consistency_matrix = computePCMMatrix(
                _loop_queue[_robot_this_th]);//, _pose_queue[_robot_this_th]);
        stringstream ss_consistency_matrix_file;
        ss_consistency_matrix_file << _pcm_matrix_folder << "/consistency_matrix" << _robot_this_th << ".clq.mtx";
        printPCMGraph(consistency_matrix, ss_consistency_matrix_file.str());
        // Compute maximum clique
        FMC::CGraphIO gio;
        gio.readGraph(ss_consistency_matrix_file.str());
        std::vector<int> max_clique_data;

        FMC::maxCliqueHeu(gio, max_clique_data);

        std::sort(max_clique_data.begin(), max_clique_data.end());

        auto loop_accept_queue_this = _loop_accept_queue.find(_robot_this_th);
        if (loop_accept_queue_this == _loop_accept_queue.end()) {
            _loop_accept_queue.emplace(std::make_pair(_robot_this_th, max_clique_data));
            return true;
        }

        if (max_clique_data == loop_accept_queue_this->second)
            return false;

        _loop_accept_queue[_robot_this_th].clear();
        _loop_accept_queue[_robot_this_th] = max_clique_data;
        return true;
    }

    inline gtsam::Pose3_ transformTo(const gtsam::Pose3_ &x, const gtsam::Pose3_ &p) {
        return gtsam::Pose3_(x, &gtsam::Pose3::transformPoseTo, p);
    }

    void gtsamExpressionGraph(string sub_dataset_dir) {
        if (_loop_accept_queue[_robot_this_th].size() < 2)
            return;

        gtsam::Vector Vector6(6);

        gtsam::Pose3 initial_pose_0, initial_pose_1;
        initial_pose_0 = std::get<0>(_pose_queue[_robot_this_th][_loop_accept_queue[_robot_this_th][
                _loop_accept_queue[_robot_this_th].size() - 1]]);
        initial_pose_1 = std::get<1>(_pose_queue[_robot_this_th][_loop_accept_queue[_robot_this_th][
                _loop_accept_queue[_robot_this_th].size() - 1]]);

        gtsam::Values initial;
        gtsam::ExpressionFactorGraph graph;

        gtsam::Pose3_ trans(0);

        initial.insert(0, initial_pose_1 * initial_pose_0.inverse());
        //initial.print();

        gtsam::Pose3 p, measurement;
        float noiseScore;

        for (auto i: _loop_accept_queue[_robot_this_th]) {
            std::tie(measurement, p, noiseScore) = _pose_queue[_robot_this_th][i];

            if (noiseScore == 0)// 0 indicates a inter robot outlier
                continue;

            gtsam::Pose3_ predicted = transformTo(trans, p);

            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
                    noiseScore;
            auto measurementNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

            // Add the Pose3 expression variable, an initial estimate, and the measurement noise.
            graph.addExpressionFactor(predicted, measurement, measurementNoise);
        }
        gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();

        gtsam::Pose3 est = result.at<gtsam::Pose3>(0);

        PointTypePose map_trans_this;

        //float x, y, z, roll, pitch, yaw;
        map_trans_this.x = est.translation().x();
        map_trans_this.y = est.translation().y();
        map_trans_this.z = est.translation().z();
        map_trans_this.roll = est.rotation().roll();
        map_trans_this.pitch = est.rotation().pitch();
        map_trans_this.yaw = est.rotation().yaw();

        auto ite = _global_map_trans.find(_robot_this_th);
        if (ite == _global_map_trans.end()) {
            std::vector<PointTypePose> tmp_pose_list;
            tmp_pose_list.push_back(map_trans_this);
            _global_map_trans.emplace(std::make_pair(_robot_this_th, tmp_pose_list));
            _global_map_trans_optimized.emplace(std::make_pair(_robot_this_th, map_trans_this));
        } else {
            _global_map_trans[_robot_this_th].push_back(map_trans_this);
            _global_map_trans_optimized[_robot_this_th] = map_trans_this;
        }

        if (_robot_this_th == 0){
            fstream stream(sub_dataset_dir + "trans2dataset1.txt", fstream::out);
            stream << "x y z roll pitch yaw" <<endl;
            stream << map_trans_this.x << " " << map_trans_this.y << " " << map_trans_this.z << " "
                   << map_trans_this.roll << " " << map_trans_this.pitch << " " << map_trans_this.yaw << endl;

            stream.close();
        }

        graph.resize(0);
    }

    void writeG2o1(const NonlinearFactorGraph &graph, const Values &estimate,
                   const std::string &filename) {
        std::fstream stream(filename.c_str(), std::fstream::out);

        // Use a lambda here to more easily modify behavior in future.
        auto index = [](gtsam::Key key) { return Symbol(key).index(); };

        // save 3D poses
        for (const auto key_value : estimate) {
            auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
            if (!p)
                continue;
            const Pose3 &pose = p->value();
            const Point3 t = pose.translation();
            const auto q = pose.rotation().toQuaternion();
            stream << "VERTEX_SE3:QUAT " << index(key_value.key) << " " << t.x() << " "
                   << t.y() << " " << t.z() << " " << q.x() << " " << q.y() << " "
                   << q.z() << " " << q.w() << endl;
        }

        // save edges (2D or 3D)
        for (const auto &factor_ : graph) {
            auto factor3D = boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(factor_);

            if (factor3D) {
                SharedNoiseModel model = factor3D->noiseModel();

                boost::shared_ptr<noiseModel::Gaussian> gaussianModel =
                        boost::dynamic_pointer_cast<noiseModel::Gaussian>(model);
//            if (!gaussianModel) {
//                model->print("model\n");
//                throw invalid_argument("writeG2o: invalid noise model!");
//            }
                if(gaussianModel) {
                    Matrix6 Info = gaussianModel->R().transpose() * gaussianModel->R();
                    const Pose3 pose3D = factor3D->measured();
                    const Point3 p = pose3D.translation();
                    const auto q = pose3D.rotation().toQuaternion();
                    stream << "EDGE_SE3:QUAT " << factor3D->key1() << " "
                           << factor3D->key2() << " " << p.x() << " " << p.y() << " "
                           << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                           << q.w();

                    Matrix6 InfoG2o = I_6x6;
                    InfoG2o.block<3, 3>(0, 0) = Info.block<3, 3>(3, 3); // cov translation
                    InfoG2o.block<3, 3>(3, 3) = Info.block<3, 3>(0, 0); // cov rotation
                    InfoG2o.block<3, 3>(0, 3) = Info.block<3, 3>(0, 3); // off diagonal
                    InfoG2o.block<3, 3>(3, 0) = Info.block<3, 3>(3, 0); // off diagonal

                    for (size_t i = 0; i < 6; i++) {
                        for (size_t j = i; j < 6; j++) {
                            stream << " " << InfoG2o(i, j);
                        }
                    }
                    stream << endl;
                }
                else{
//                boost::shared_ptr<noiseModel::mEstimator::Cauchy> robustModel =
//                        boost::dynamic_pointer_cast<noiseModel::mEstimator::Cauchy>(model);
//                Matrix6 Info = gaussianModel->R().transpose() * gaussianModel->R();
                    const Pose3 pose3D = factor3D->measured();
                    const Point3 p = pose3D.translation();
                    const auto q = pose3D.rotation().toQuaternion();
                    stream << "EDGE_SE3:QUAT " << index(factor3D->key1()) << " "
                           << index(factor3D->key2()) << " " << p.x() << " " << p.y() << " "
                           << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                           << q.w();

//                Matrix6 InfoG2o = I_6x6;
//                InfoG2o.block<3, 3>(0, 0) = Info.block<3, 3>(3, 3); // cov translation
//                InfoG2o.block<3, 3>(3, 3) = Info.block<3, 3>(0, 0); // cov rotation
//                InfoG2o.block<3, 3>(0, 3) = Info.block<3, 3>(0, 3); // off diagonal
//                InfoG2o.block<3, 3>(3, 0) = Info.block<3, 3>(3, 0); // off diagonal

                    for (size_t i = 0; i < 6; i++) {
                        for (size_t j = i; j < 6; j++) {
                            stream << " " << 0.5;
                        }
                    }
                    stream << endl;

                }

            }
        }
        stream.close();
    }


    void run(){

        for (int i=0; i<_sub_dataset_num; i++){
            stringstream ss;
            ss << _dataset_dir << i+1 << "/";
            //read poses & factors
            string sub_dataset_dir = ss.str();
            cout<< "Processing " << sub_dataset_dir << "..."<<endl;
            GraphAndValues gv = readG2o(sub_dataset_dir + "graph.g2o",true);
            Values v = * gv.second;
            //read pcd files
            KeyVector keys = v.keys();
            for (auto key : keys){
                if (key % 1500 == 0) {
                    cout << "key: " << key << endl;
                }

                stringstream ss0;
                ss0 << sub_dataset_dir << "Scans/" << std::setw(6) << std::setfill('0') << key << ".pcd";
                string pcl_bin = ss0.str();
                pcl::PCDReader reader;

                pcl::PointCloud<PointType>::Ptr cloud;
                cloud.reset(new pcl::PointCloud<PointType>());
                reader.read <PointType> ( pcl_bin,  * cloud);

                ScanContextBin bin = _scan_context_factory->ptcloud2bin(cloud);
                stringstream ss1;
                ss1 << "data" << i;
                bin.robotname = ss1.str();
                bin.time = key;

                Pose3 pose_this = v.at<Pose3>(key);
                bin.pose.x = pose_this.x();
                bin.pose.y = pose_this.y();
                bin.pose.z = pose_this.z();
                bin.pose.roll = pose_this.rotation().roll();
                bin.pose.pitch = pose_this.rotation().pitch();
                bin.pose.yaw = pose_this.rotation().yaw();
                bin.pose.intensity = key;

                buildKDTree(bin);

                KNNSearch(bin, i);

                if( !getInitialGuesses(bin, i))
                    continue;

                if(!incrementalPCM())
                    continue;

                gtsamExpressionGraph(sub_dataset_dir);

            }
            NonlinearFactorGraph graph_bin;

            gtsam::Vector Vector6(6);
            Vector6 << 2,2,2,2,2,2;
            auto odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

            for (auto ite_0 : _loop_queue){
                auto vector_this = ite_0.second;

                for (auto tuple_this : vector_this){
                    ScanContextBin b0 = _bin_with_id[get<0>(tuple_this)];
                    ScanContextBin b1 = _bin_with_id[get<1>(tuple_this)];
                    char c0 = robotID2Number(b0.robotname) +'a';
                    char c1 = robotID2Number(b1.robotname) +'a';

                    Symbol key0 = symbol(c0, int(b0.pose.intensity));
                    Symbol key1 = symbol(c1, int(b1.pose.intensity));
                    graph_bin.add(BetweenFactor<Pose3>(key0, key1, get<2>(tuple_this), odometryNoise) );

                }
            }
            writeG2o1(graph_bin, Values(),sub_dataset_dir + "inter_graph.g2o");

            graph_bin.resize(0);

            _robot_received_list.clear();
            _global_map_trans.clear();
            _global_map_trans_optimized.clear();

            _pose_queue.clear();
            _loop_queue.clear();

        }
    }

};

int main(int argc, char** argv) {
//    ros::init(argc,argv,"lio_connector");string dataset_dir =
    LioConnector lc("/media/External-nvme/Michigan/data/data-", 3);
    lc.run();

}