#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils.h"
#include "types.h"
#include <fstream>
#include <thread>

#include "phtree-cpp/include/phtree/phtree.h"

#include "ikd_tree/ikd_tree.h"

using namespace improbable::phtree;

using PointType = ikdTree_PointType;
using PointVector = KD_TREE<PointType>::PointVector;

#define USE_OMP 0

#define PHTREE 0
#define IKDTREE 1

#define METHOD PHTREE
//#define METHOD IKDTREE

// Seems that the IKD-Tree needs to be defined globally https://github.com/hku-mars/ikd-Tree/issues/8#issuecomment-877589515
#if METHOD == IKDTREE
KD_TREE<ikdTree_PointType> ikd_tree;
#endif



// Main
int main(int argc, char** argv)
{
    // Hardcoded simulation parameters
    const int kTrajectoryLength = 5000;
    const int kNumPoints = 10000000;
    const int kPerStepNumPoints = 20000;
    const double kLidarRange = 100.0;
    const int kNumQueriesPerStep = 8000;

    const int kNumSteps = kNumPoints / kPerStepNumPoints;
    const double kDistInterval = ((double)kTrajectoryLength) / kNumSteps;

    // Create a PH-Tree if needed
    #if METHOD == PHTREE
    // Create a PH-Tree with 3 dimensions and integer values
    PhTreeD<3, int> tree;
    #endif

    // Store times
    std::vector<double> time_insertions(kNumSteps);
    std::vector<double> time_queries(kNumSteps);


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
            #endif
        }
        time_queries[step] = sw.stop();
        // Little sanity check and forcing compiler to not optimize away the queries
        std::cout << "Average query point: " << query_pts_res.colwise().mean() << std::endl;
        sw.print("Query time for step " + std::to_string(step) + ": ");
    }

    // Write the times to file
    #if METHOD == PHTREE
    std::string method_name = "phtree";
    #elif METHOD == IKDTREE
    std::string method_name = "ikdtree";
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
        file << time_insertions[i] << "," << time_queries[i] << "\n";
    }
    file.close();
    return 0;
}
