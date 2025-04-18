/*
File: JH_AP_project_3.cpp
Authors: 
    John Hohman
    Austin Pennington
Created: 01-23-2025
Last Modified: 02-15-2025

Description:
    Implementation of optimization and integration algorithms:
        1. MADS (Mesh Adaptive Direct Search) for finding global optima
        2. Parallel Monte Carlo integration using OpenMP

Dependencies:
    OpenMP for parallel processing
    C++20 or later

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
#include <format>
#include <functional>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <system_error>
#include <fstream>
#include <omp.h>

std::vector<std::vector<double>> generateScouts(std::vector<double> origin, int scouts, double distance, unsigned int seed = 10110101){
    //mersene twister random number generator
    std::mt19937 gen{seed}; 
    std::normal_distribution d{0.0, distance};

    int dims = origin.size(); 
    std::vector<std::vector<double>> coordinate_set;

    for(int i = 0; i < scouts; i++){
        std::vector<double> coordinate;
        for(int j = 0; j < dims; j++){
            coordinate[j] = origin[j] + d(gen);
        }
        coordinate_set.push_back(coordinate);
    }

    return coordinate_set;
}

// A single macro for doing numeric mins and maxs
#define NUMERIC_MAX_MIN(a, b, use_max) ((use_max) ? ((a) > (b) ? (a) : (b)) : ((a) < (b) ? (a) : (b)))

double MADS(std::function<double(std::vector<double>)> objective, 
            bool mode, 
            size_t itmax = 100, double delta = 1, int scouts = 20) {
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

    //@TODO calculate number of dimensions
    double delta = 1.0;
    std::vector<double> origin;
    std::vector<double> best_coordinate = origin;
    std::vector<std::vector<double>> coordinate_set = generateScouts(origin, scouts, delta);
    
    double best_optima = 0;
    double new_optima = 0;

    unsigned int seed = 1;

    for(int it = 0; it < itmax; it++){
        
        best_optima = 0;
        new_optima = 0;

        for(int j = 0; j < scouts; j++){
            //define cords via circle
            double curr_evaluation = objective(coordinate_set[j]);
            
            new_optima = NUMERIC_MAX_MIN(best_optima, curr_evaluation, mode);

            if(new_optima > best_optima){
                best_coordinate = coordinate_set[j];
                best_optima = new_optima;
            }
        }
        // @TODO actually implement adaptivity
        delta *= 0.95;
        coordinate_set = generateScouts(best_coordinate, scouts, delta, seed);
        seed++;
    }

    return best_optima;
}


// A custom error to indicate that user thread count input is either over or under the system bounds
class ThreadCountOutOfBoundException : public std::runtime_error, public std::nested_exception{
public:
    explicit ThreadCountOutOfBoundException(const char* message)
        : std::runtime_error(message) {}
};

// A small helper function to handle errors with setting thread count
void set_thread_count(int thread_count){
    try{
        if(thread_count < 1){
            throw ThreadCountOutOfBoundException("Thread count must be at least 1");
        }

        int max_threads = omp_get_thread_limit();

        if(thread_count > max_threads){
            throw ThreadCountOutOfBoundException(std::format("Thread count {} exceeds system maximum of {}", thread_count, max_threads));
        }

        omp_set_num_threads(thread_count);

    }catch(const ThreadCountOutOfBoundException& e){
        fprintf(stderr, "ThreadCountOutOfBoundException: %s\n", e.what());

        try{
            std::rethrow_if_nested(e);
        }catch(const std::exception& nested){
            fprintf(stderr, "Caused by: %s\n", nested.what());
        }

        throw;
    }catch(const std::exception& e){
        fprintf(stderr, "Unexpected Error: %s\n", e.what());
        throw;
    }
}


double Monte_Carlo_Integration(std::function<double(std::vector<double>)> objective,
                    double min, double max, 
                    std::vector<double> lower_bound, std::vector<double> upper_bound, 
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

lower_bound : std::vector<double>
    Lower Boundarys

upper_bound : std::vector<double>
    Upper Boundarys

itmax : size_t 
    Amount of sample points we'll create around us to determine which step gets us closer to our objective

thread_count : unsigned short
    Amount of Threads We'll Create

Returns:
    double: integral approximation

Raises:
    ThreadCountOutOfBoundsException
        - Thread Count < 1 
        - Thread Count > MAX NUMBER OF THREADS ALLOWED ON LINUX (e.g. cat /proc/sys/kernel/threads-max)

*/          

    set_thread_count(thread_count);

    double area = 0;
    // vector size equals the number of parallel partitions
    std::vector<double> local_area(thread_count, 0.0);

    // @TODO Implement montecarlo integration per thread

    //Create array of vectors/points
    
    #pragma omp parallel for
    {
        for(size_t it = 0; it < itmax; it++ ) {
            //@TODO implement n-dimensional polymorphism
            std::mt19937 gen{seed}; // @TODO Generate point (which is a vector)   
            std::uniform_real_distribution(lower_bound, upper_bound);
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
    auto objective1 = [](std::vector<double> vars) -> double {
        return {vars[0] * vars[0] + vars[1] * vars[1]};
    };

    // For x + 2
    auto objective2 = [](std::vector<double> vars) -> double {
        return {vars[0] + 2};
    };

    // For w + x + y + z
    auto objective3 = [](std::vector<double> vars) -> double {
        return {vars[0] + vars[1] + vars[2] + vars[3]};
    };

    // Define upper and lower bounds of the integral
    std::vector<double> upper_bound =  {2};
    std::vector<double> lower_bound = {-2};


    // Generating both the maximum and minimum parameters for Monte Carlo 
    double max = MADS(objective2, true);
    double min = MADS(objective2, false);

    double area = Monte_Carlo_Integration(objective2, min, max, lower_bound, upper_bound);

    std::cout << "The integral of x + 2 from " << lower_bound[0] << " to " << upper_bound[0] << " approximately equals " << area << "\n";

    return 0;
}