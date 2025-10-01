# Import necessary libraries
import cv2
import numpy as np
import pickle
import os
import ot
import scipy

# Define distance and similarity functions
def euclidean_distance(h1, h2):
    return np.linalg.norm(h1 - h2, ord=2)

def l1_distance(h1, h2):
    return np.linalg.norm(h1 - h2, ord=1)

def x2_distance(h1, h2):
    return np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))

def hist_intersection(h1, h2):
    # this similarity gives != 0 even when the histograms are identical 
    return np.sum(np.minimum(h1, h2))

def hellinger_similarity(h1, h2):
    # this similarity gives != 0 even when the histograms are identical 
    return np.sum(np.sqrt(h1*h2))

def kl_divergence(h1, h2):
    return sum(h1[i] * np.log(h1[i]/h2[i]) for i in range(len(h1)))

def jensen_shannon_divergence(h1, h2):
    # https://medium.com/data-science/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6
    m = 0.5 * (h1 + h2)
    return 0.5 * (kl_divergence(h1, m) + kl_divergence(h2, m))


def earth_movers_distance(h1, h2, bin_locations):
    # h1, h2: Histograms to compare
    # bin locations: Locations of the bins, the "values" of the bins

    return scipy.stats.wasserstein_distance(bin_locations, bin_locations, h1, h2) # SCIPY IMPLEMENTATION


def quadratic_form_distance(h1, h2,A):
    # h1, h2: Histograms to compare
    # A: Similarity matrix
    diff = h1 - h2
    return np.sqrt(diff.T @ A @ diff)

def emd (a,b):
    # https://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance
    earth = 0
    earth1 = 0
    diff = 0
    s= len(a)
    su = []
    diff_array = []
    for i in range (0,s):
        diff = a[i]-b[i]
        diff_array.append(diff)
        diff = 0
    for j in range (0,s):
        earth = (earth + diff_array[j])
        earth1= abs(earth)
        su.append(earth1)
    emd_output = sum(su)/(s-1)
    return emd_output

def emd_ot(h1,h2,bin_locations):
    # h1, h2: Histograms to compare
    # bin locations: Locations of the bins, the "values" of the bins
    n_bins = len(h1)
    # Cost matrix: distance between bin i and bin j
    cost_matrix = ot.dist(bin_locations.reshape((n_bins, 1)), bin_locations.reshape((n_bins, 1)))

    ot_emd = ot.emd2(h1, h2, cost_matrix)

    return ot_emd

if __name__ == "__main__":
    hist1 = np.array([1]*4)
    hist2 = np.array([2]*4)

    print("Euclidean Distance:", euclidean_distance(hist1, hist2))
    print("L1 Distance:", l1_distance(hist1, hist2))
    print("X-Squared Distance:", x2_distance(hist1, hist2))
    print("Histogram Intersection:", hist_intersection(hist1, hist2))
    print("Hellinger Similarity:", hellinger_similarity(hist1, hist2))
    print("KL Divergence:", kl_divergence(hist1, hist2))
    print("Jensen-Shannon Divergence:", jensen_shannon_divergence(hist1, hist2))
