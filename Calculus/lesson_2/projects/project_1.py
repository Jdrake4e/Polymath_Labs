import csv
import numpy as np

def f_1(t):
    return np.sin(t) + t
def f_2(t):
    return np.cos(t)
def f_3(t):
    return np.log(t)

if __name__ == "__main__":
    output_file = "project_1_data.csv"
    times = np.linspace(0.0001, 10, 1000)

    # Write to the CSV file
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["t", "f_1(x)", "f_2(x)", "f_3(x)"])
        # Write the data rows
        for t in times:
            writer.writerow([t, f_1(t), f_2(t), f_3(t)])
            
    '''
        Goal 1: Implement the Trapezoid Method for solving ordinary differential equations
        Goal 2: apporixmate the true function
    '''