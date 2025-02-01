/*
File: JH_AP_project_3.cpp
Authors: 
    John Hohman
    Austin Pennington
Created: 01-23-2025
Last Modified: 01-31-2025

Description:
    Implementation of optimization and integration algorithms:
        1. MADS (Mesh Adaptive Direct Search) for finding global optima
        2. Parallel Monte Carlo integration using OpenMP

Dependencies:
    OpenMP for parallel processing
    C++11 or later

Usage:
    Compile with OpenMP support:
    g++ -fopenmp optimization.cpp -o optimization

 Notes:
    MADS implementation supports both minimization and maximization
    Monte Carlo integration uses parallel processing for performance
    Supports multivariate objective functions

Sources:
    Monte Carlo Integration: 
        https://www.youtube.com/watch?v=8276ZswRw7M

    MADS:
        https://search.r-project.org/CRAN/refmans/dfoptim/html/mads.html
        https://www.researchgate.net/publication/220133585_Mesh_Adaptive_Direct_Search_Algorithms_for_Constrained_Optimization
        https://apps.dtic.mil/sti/tr/pdf/ADA444692.pdf

Copyright (c) 2025. Educational Purposes only.
*/

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <cstdlib>
#include <omp.h>

double MADS(std::function<std::vector<double>(std::vector<double>)> objective, 
                         bool mode, size_t itmax = 100, double delta = 1) {
/*
Summary
-------

MADS (Mesh Adaptive Direct Search) is an optimization algorithm that uses an adaptive mesh to find global minima or maxima of multivariate functions. 
It's particularly effective for non-smooth or discontinuous functions where gradient-based methods might fail.

Parameters
----------
objective : function<vector<double>(vector<double>)>
    The function to find the min and max from

mode : bool
    true = Max. Finding Mode
    false = Min. Finding Mode

itmax : size_t 
    maximum number of steps we'll take towards our objective

thread_count : int
    Amount of Threads

Returns:
    double

Raises 
*/
    return {};
}


double Monte_Carlo_Integration(std::function<std::vector<double>(std::vector<double>)> objective,
                    double min, double max, 
                    double lower_bound, double upper_bound, 
                    size_t itmax = 1000, unsigned short thread_count = 1){
/* 
Summary
-------

Implements parallel Monte Carlo integration to approximate the definite integral of a function by randomly sampling points and determining the ratio of points bounded by the curve.

Parameters
----------
min : double
    Minimum calulated through the MADS function

max : double
    Maximum calculated through the MADS function

lower_bound : double
    Lower Boundary

upper_bound : double
    Upper Boundary

itmax : size_t 
    Amount of sample points we'll create around us to determine which step gets us closer to our objective

thread_count : unsigned short
    Amount of Threads We'll Create

Returns:
    double: integral approximation

Raises:
    ThreadCountOutOfBoundsException
        - Thread Count < 1 
        - Thread Count > MAX NUMBER OF THREADS ALLOWED ON LINUX (e.g. 513510 || cat /proc/sys/kernel/threads-max)

*/                       

    // @TODO add try catch for thread count
    omp_set_num_threads(thread_count);

    double area = 0;
    // vector size equals the number of parallel partitions
    std::vector<double> local_area(thread_count, 0.0);

    // @TODO Implement montecarlo integration per partition
    #pragma omp parallel for
    {
        for(size_t i = 0; i < itmax; i++ ) {
            
        }
    }

    // @TODO Implement ratio math
    #pragma omp parallel reduction(+:area)
    {
        for(size_t i = 0; i < local_area.size(); i++){
            area += local_area[i];
        }
    }
    
    return area;
}

int main(){

    //lambdas

    // For x^2 + y^2
    auto objective1 = [](std::vector<double> vars) -> std::vector<double> {
        return {vars[0] * vars[0] + vars[1] * vars[1]};
    };

    // For x + 2
    auto objective2 = [](std::vector<double> vars) -> std::vector<double> {
        return {vars[0] + 2};
    };

    // For w + x + y + z
    auto objective3 = [](std::vector<double> vars) -> std::vector<double> {
        return {vars[0] + vars[1] + vars[2] + vars[3]};
    };

    // Define upper and lower bounds of the integral
    double upper_bound =  2;
    double lower_bound = -2;


    // Generating both the maximum and minimum parameters for Monte Carlo 
    double max = MADS(objective2, true);
    double min = MADS(objective2, false);

    double area = Monte_Carlo_Integration(objective2, min, max, lower_bound, upper_bound);

    std::cout << "The integral of x + 2 from " << lower_bound << " to " << upper_bound << " approximately equals " << area << "\n";

    return 0;
}