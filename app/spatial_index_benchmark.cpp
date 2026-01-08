#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils.h"
#include "types.h"
#include <fstream>
#include <thread>

#include "phtree-cpp/include/phtree/phtree.h"

#include "ikd_tree/ikd_tree.h"

#include "i-octree/octree2/Octree.h"

using namespace improbable::phtree;

using PointType = ikdTree_PointType;
using PointVector = KD_TREE<PointType>::PointVector;

#define USE_OMP 0

#define PHTREE 0
#define IKDTREE 1
#define IOCTREE 2

//#define METHOD PHTREE
//#define METHOD IKDTREE
#define METHOD IOCTREE

// Seems that the IKD-Tree needs to be defined globally https://github.com/hku-mars/ikd-Tree/issues/8#issuecomment-877589515
#if METHOD == IKDTREE
KD_TREE<ikdTree_PointType> ikd_tree;
#endif

struct CustomPoint
{
    double x, y, z;
};

// Main
int main(int argc, char** argv)
{
    // Hardcoded simulation parameters
    const int kTrajectoryLength = 5000;
    const int kNumPoints = 10000000;
    const int kPerStepNumPoints = 20000;
    const double kLidarRange = 100.0;
    const int kNumQueriesPerStep = 8000;
    const int kRemoveEveryNPts = 8;

    const int kNumSteps = kNumPoints / kPerStepNumPoints;
    const double kDistInterval = ((double)kTrajectoryLength) / kNumSteps;

    // Create a PH-Tree if needed
    #if METHOD == PHTREE
    // Create a PH-Tree with 3 dimensions and integer values
    PhTreeD<3, int> tree;
    #elif METHOD == IOCTREE
    // Create an i-Octree
    thuni::Octree ioctree;
    #endif

    // Store times
    std::vector<double> time_insertions(kNumSteps);
    std::vector<double> time_queries(kNumSteps);
    std::vector<double> time_removals(kNumSteps);


    // Loop over the trajectory
    for(int step = 0; step < kNumSteps; step++)
    {
        std::cout << "Step " << step << "/" << kNumSteps << std::endl;

        // Current position along a diagonal line
        Vec3 position(step * kDistInterval * std::sqrt(2)/2.0, step * kDistInterval * std::sqrt(2)/2.0, 0.0);

        // Create N random points around the current position
        MatXRowMajor pts = MatXRowMajor::Random(kPerStepNumPoints, 3) * kLidarRange;
        pts.rowwise() += position.transpose();

        // Create stopwatch to measure insertion time
        StopWatch sw;
        sw.start();

        #if METHOD == PHTREE
        // Insert points one-by-one in the PH-Tree
        int offset = step * kPerStepNumPoints;
        for (int i = 0; i < kPerStepNumPoints; i++)
        {
            PhPointD<3> pt({pts(i, 0), pts(i, 1), pts(i, 2)});
            tree.emplace(pt, i + offset);
        }
        #elif METHOD == IKDTREE
        // Convert points to ikdTree_PointType (could be optimized to avoid this copy but negligible time here)
        std::vector<ikdTree_PointType, Eigen::aligned_allocator<ikdTree_PointType>> ikd_pts(kPerStepNumPoints);
        for (int i = 0; i < kPerStepNumPoints; i++)
        {
            ikd_pts[i] = ikdTree_PointType(pts(i, 0), pts(i, 1), pts(i, 2));
        }
        if(step == 0)
        {
            ikd_tree.Build(ikd_pts);
        }
        else
        {
            ikd_tree.Add_Points(ikd_pts, false);
        }
        #elif METHOD == IOCTREE
        // Convert points to CustomPoint
        std::vector<CustomPoint, Eigen::aligned_allocator<CustomPoint>> ioctree_pts(kPerStepNumPoints);
        for (int i = 0; i < kPerStepNumPoints; i++)
        {
            ioctree_pts[i] = CustomPoint{pts(i, 0), pts(i, 1), pts(i, 2)};
        }
        if(step == 0)
        {
            ioctree.initialize(ioctree_pts);
        }
        else
        {
            ioctree.update(ioctree_pts);
        }
        //// Convert points to CustomPoint
        //for (int i = 0; i < kPerStepNumPoints; i++)
        //{
        //    std::vector<CustomPoint, Eigen::aligned_allocator<CustomPoint>> ioctree_pts(1);
        //    ioctree_pts[0] = CustomPoint{pts(i, 0), pts(i, 1), pts(i, 2)};
        //    if(step == 0 && i == 0)
        //    {
        //        ioctree.initialize(ioctree_pts);
        //    }
        //    else
        //    {
        //        ioctree.update(ioctree_pts);
        //    }
        //}
        #endif

        // Save and display insertion time
        time_insertions[step] = sw.stop();
        sw.print("Insertion time for step " + std::to_string(step) + ": ");


        // Create query points around the current position
        MatXRowMajor query_pts = MatXRowMajor::Random(kNumQueriesPerStep, 3) * kLidarRange;
        query_pts.rowwise() += position.transpose();

        #if METHOD == IKDTREE
        // Sleep for 50ms (added based on https://github.com/hku-mars/ikd-Tree/issues/20)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        #endif

        // Reset the stopwatch
        sw.reset();
        sw.start();

        // Query the nearest neighbor for each query point
        MatX query_pts_res(kNumQueriesPerStep, 3);
        #if USE_OMP
        #pragma omp parallel for num_threads(8)
        #endif
        for (int i = 0; i < kNumQueriesPerStep; i++)
        {
            Vec3 query_pt = query_pts.row(i);

            #if METHOD == PHTREE
            for(auto nn=tree.begin_knn_query(1,{query_pt[0], query_pt[1], query_pt[2]}, DistanceEuclidean<3>()); nn!=tree.end(); ++nn)
            {
                auto closest_pt = nn.first();
                query_pts_res(i, 0) = closest_pt[0];
                query_pts_res(i, 1) = closest_pt[1];
                query_pts_res(i, 2) = closest_pt[2];
                break;
            }
            #elif METHOD == IKDTREE
            KD_TREE<ikdTree_PointType>::PointVector ret_pts;
            std::vector<float> dists;
            ikdTree_PointType ikd_query_pt(query_pt[0], query_pt[1], query_pt[2]);
            ikd_tree.Nearest_Search(ikd_query_pt, 1, ret_pts, dists);
            for (auto& pt : ret_pts)
            {
                query_pts_res(i, 0) = pt.x;
                query_pts_res(i, 1) = pt.y;
                query_pts_res(i, 2) = pt.z;
                break;
            }
            #elif METHOD == IOCTREE
            std::vector<size_t> results;
            std::vector<double> dist_results;
            CustomPoint ioctree_query_pt{query_pt[0], query_pt[1], query_pt[2]};
            ioctree.knnNeighbors(ioctree_query_pt, 1, results, dist_results);
            if(!results.empty())
            {
                size_t idx = results[0];
                query_pts_res(i, 0) = ioctree_query_pt.x;
                query_pts_res(i, 1) = ioctree_query_pt.y;
                query_pts_res(i, 2) = ioctree_query_pt.z;
            }
            else
            {
                throw std::runtime_error("No nearest neighbor found in i-Octree");
            }
            #endif
        }
        time_queries[step] = sw.stop();
        // Little sanity check and forcing compiler to not optimize away the queries
        std::cout << "Average query point: " << query_pts_res.colwise().mean() << std::endl;
        sw.print("Query time for step " + std::to_string(step) + ": ");


        // Reset the stopwatch
        sw.reset();
        sw.start();
        // Remove some points random
        std::vector<Vec3> points_to_remove;
        for(int i = 0; i < kPerStepNumPoints; i += kRemoveEveryNPts)
        {
            Vec3 pt = pts.row(i);
            points_to_remove.push_back(pt);
        }
        #if METHOD == PHTREE
        for(const auto& pt : points_to_remove)
        {
            PhPointD<3> ph_pt({pt[0], pt[1], pt[2]});
            tree.erase(ph_pt);
        }
        #elif METHOD == IKDTREE
        std::cout << "NOT IMPLEMENTED: Removal in IKD-Tree" << std::endl;
        #elif METHOD == IOCTREE
        const double quantum = 1e-6;
        for(const auto& pt : points_to_remove)
        {
            thuni::BoxDeleteType box;
            box.min[0] = pt[0] - quantum;
            box.min[1] = pt[1] - quantum;
            box.min[2] = pt[2] - quantum;
            box.max[0] = pt[0] + quantum;
            box.max[1] = pt[1] + quantum;
            box.max[2] = pt[2] + quantum;
            ioctree.boxWiseDelete(box, true);
        }
        #endif
        time_removals[step] = sw.stop();
        sw.print("Removal time for step " + std::to_string(step) + ": ");




    }

    // Write the times to file
    #if METHOD == PHTREE
    std::string method_name = "phtree";
    #elif METHOD == IKDTREE
    std::string method_name = "ikdtree";
    #elif METHOD == IOCTREE
    std::string method_name = "ioctree";
    #endif

    #if USE_OMP
    method_name += "_omp";
    #endif

    // Create a csv file with the times
    std::string filename = "spatial_indexing_times_" + method_name + ".csv";
    std::ofstream file(filename);
    file << "InsertionTime_ms,QueryTime_ms\n";
    for (size_t i = 0; i < time_insertions.size(); i++)
    {
        file << time_insertions[i] << "," << time_queries[i] << "," << time_removals[i] << "\n";
    }
    file.close();

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    return 0;
}
