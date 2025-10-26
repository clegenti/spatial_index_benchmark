import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

kPlotOmp = False

methods = {
    "PH-Tree" : ["build/spatial_indexing_times_phtree", "blue"],
    "ikd-Tree" : ["build/spatial_indexing_times_ikdtree", "orange"],
}

fig, ax = plt.subplots(1,2, figsize=(8, 3))
for method, filebase in methods.items():
    try:
        data_single_core = np.loadtxt(filebase[0] + ".csv", delimiter=',', skiprows=1)

        num_pts = (np.arange(data_single_core.shape[0])) * 20000
        ax[0].plot(num_pts, data_single_core[:,0], c=filebase[1], label=method + " (avg: " + np.mean(data_single_core[:,0]).round(2).__str__() + "ms)")
        ax[1].plot(num_pts, data_single_core[:,1], c=filebase[1], label=method + " (avg: " + np.mean(data_single_core[:,1]).round(2).__str__() + "ms)")
    except Exception as e:
        print("Could not load data for method " + method + ": " + str(e))
        continue
    if kPlotOmp:
        try:
            num_pts = (np.arange(data_single_core.shape[0])) * 20000
            data_omp = np.loadtxt(filebase[0] + "_omp.csv", delimiter=',', skiprows=1)
            ax[1].plot(num_pts, data_omp[:,1], c=filebase[1], linestyle='dashed', label=method + " 8 threads (avg: " + np.mean(data_omp[:,1]).round(2).__str__() + "ms)")
        except Exception as e:
            print("Could not load OMP data for method " + method + ": " + str(e))
            continue

x_labels = ['0', '2M', '4M', '6M', '8M', '10M']
x_ticks = [0, 2000000, 4000000, 6000000, 8000000, 10000000]
for a in ax:
    a.set_xticks(x_ticks)
    a.set_xticklabels(x_labels)
    a.set_xlabel("Number of points in map")
    a.set_ylabel("Time (ms)")
    a.legend(loc='upper left', framealpha=1.0)

ax[0].set_title("Time for 20k insertions")
ax[1].set_title("Time for 8k queries")


plt.subplots_adjust(top=0.766,
bottom=0.194,
left=0.088,
right=0.981,
hspace=0.2,
wspace=0.22)
plt.suptitle("PH-Tree vs ikd-Tree timing (incrementally building and querying a 10M-point map)")
if kPlotOmp:
    plt.savefig("timing_comparison_omp.png", dpi=300)
    plt.savefig("timing_comparison_omp.pdf")
else:
    plt.savefig("timing_comparison.png", dpi=300)
    plt.savefig("timing_comparison.pdf")

plt.show()