#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <functional>
#include <iomanip> //this library allows for seting precision

double solve_newton(std::function<double(double,double)> func, 
                    double t_next, double y_curr, double h,
                    double tol = 1E-6, int it_max = 100){

    double y_guess = y_curr;

    for(int i = 0; i < it_max; i++){
        double F = y_guess - y_curr - h * func(t_next, y_guess);
        
        // Finite Difference Approximation
        double eps = 1E-7;
        double dF = 1 - h * (func(t_next, y_guess+eps)-func(t_next, y_guess))/eps;

        double y_new = y_guess - F/dF;

        // check convergence
        if(std::abs(y_new - y_guess) < tol){
            return y_new;
        }
        y_guess = y_new;
    }

    //no convergence
    return y_guess;
}


std::vector<double> trapazoid_method(std::function<double(double,double)> func,
                                    double y_0, const std::vector<double>& time_points){
    std::vector<double> y_values;
    y_values.push_back(y_0);

    for (size_t i = 1; i < time_points.size(); i++){
        double h = time_points[i] - time_points[i-1];
        double y_prev = y_values.back();
        double t_next = time_points[i];
        double t_prev = time_points[i-1];

        
        // solve explicit eulers
        double y_guess = y_prev + h * func(t_prev, y_prev);

        auto trapazoid_func = [&](double t, double y){
            return (y-y_prev - (h/2.0) * (func(t_prev, y_prev) + func(t, y)));
        };

        // solve implcit eulers using newtons method
        double y_next = solve_newton(trapazoid_func, t_next, y_guess, h);

        y_values.push_back(y_next);
    }
    return y_values;
};

int main(){

    // Declare variables
    std::fstream fin;
    std::vector<double> timestep;
    std::vector<double> f_1;
    std::vector<double> f_2;
    std::vector<double> f_3;

    // file readin
    fin.open("project_1_data.csv", std::ios::in);
    if (!fin.is_open()){
        std::cerr << "Error: Could not open file" << std::endl;
        return 1;
    }

    std::cout << "File opened successfully" << std::endl;

    std::string line;
    int row_count = 0;

    //skip header
    std::getline(fin, line);
    std::cout << "Header: " << line << "\n";

    while(std::getline(fin, line)){
        
        row_count++;
        // std::cout << "Reading line " << row_count << ": " << line << "\n";

        std::stringstream str(line);
        std::string value;

        // readin each line
        try{
            if(std::getline(str, value, ',')){
                timestep.push_back(std::stod(value));
            }
            if(std::getline(str, value, ',')){
                f_1.push_back(std::stod(value));
            }
            if(std::getline(str, value, ',')){
                f_2.push_back(std::stod(value));
            }
            if(std::getline(str, value, ',')){
                f_3.push_back(std::stod(value));
            }

        } catch (const std::exception& e) {
            std::cerr << "Error converting value on line " << row_count << ": " << e.what() << std::endl;
            std::cerr << "Full line content: " << line << std::endl;
        }

        // go through first 5 rows for debug only
        // if (row_count >= 5) break;
    }

    fin.close();

    std::cout << "Number of rows read: " << timestep.size() << std::endl;

    // define functions

    auto func_1 = [](double t, double y) {return (std::cos(y) + 1.0); };

    auto func_2 = [](double t, double y) {return (-1.0*std::sin(y)); };

    auto func_3 = [](double t, double y) {return (1.0/t); };

    //run trapazoid method on each function

    double y0_1 = f_1[0];
    double y0_2 = f_2[0];
    double y0_3 = f_3[0];
    std::vector<double> solution_1 = trapazoid_method(func_1, y0_1, timestep);
    std::vector<double> solution_2 = trapazoid_method(func_2, y0_2, timestep);
    std::vector<double> solution_3 = trapazoid_method(func_3, y0_3, timestep);

    // Write results csv and error totals from actual to do error analysis

    for(size_t  i = 0; i < 5; i++){
        std::cout << "Actual: " << f_1[i];
        std::cout << " | Approx :" << solution_1[i];
        std::cout << " | Relative Error :" << std::abs(f_1[i] - solution_1[i])/std::abs(f_1[i]) << "\n";
    }

    for(size_t  i = 0; i < 5; i++){
        std::cout << "Actual: " << f_2[i];
        std::cout << " | Approx :" << solution_2[i];
        std::cout << " | Relative Error :" << std::abs(f_2[i] - solution_2[i])/std::abs(f_2[i]) << "\n";
    }

    for(size_t  i = 0; i < 5; i++){
        std::cout << "Actual: " << f_3[i];
        std::cout << " | Approx :" << solution_3[i];
        std::cout << " | Relative Error :" << std::abs(f_3[i] - solution_3[i])/std::abs(f_3[i]) << "\n";
    }

    // File output
    std::ofstream outFile("john_hohman_project_2_results.csv");

    if(!outFile.is_open()){
    std::cerr << "Error: Could not create output file" << std::endl;
    return 1;
    }

    // write header
    outFile << "Time,F1_Actual,F1_Approximation,F1_RelativeError,";
    outFile << "F2_Actual,F2_Approximation,F2_RelativeError,";
    outFile << "F3_Actual,F3_Approximation,F3_RelativeError\n";

    outFile << std::fixed << std::setprecision(6);
    for(size_t i = 0; i < timestep.size(); i++){
        outFile << timestep[i] << ",";

        outFile << f_1[i] << "," << solution_1[i] << ",";
        outFile << std::abs(f_1[i] - solution_1[i])/std::abs(f_1[i]) << ",";

        outFile << f_2[i] << "," << solution_2[i] << ",";
        outFile << std::abs(f_2[i] - solution_2[i])/std::abs(f_2[i]) << ",";

        outFile << f_3[i] << "," << solution_3[i] << ",";
        outFile << std::abs(f_3[i] - solution_3[i])/std::abs(f_3[i]) << "\n";
    }

    outFile.close();
    return 0;
}