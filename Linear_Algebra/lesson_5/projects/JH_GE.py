import numpy as np


#TODO add response when there are no solutions or infinitely many solutions
# The last column is the z in w+x+y = z when made into an Augmented matrix
def guassian_elimination(system_of_equations: np.array):
    
    #number of cols
    n = system_of_equations.shape[1]
    #num of rows
    m = system_of_equations.shape[0]
    
    row_counter = 0
    for i in range(0,min(n,m)):
        col_counter = 0
        for k in range(0, min(n,m)):
            system_of_equations[k,:] = system_of_equations[k,:]/system_of_equations[k, col_counter]
            col_counter += 1
        for j in range(i+1, min(n,m)):
            system_of_equations[j,:] = system_of_equations[j,:] - system_of_equations[j, row_counter] * system_of_equations[i, :]
        row_counter += 1
    
    row_counter = min(m,n)-1
    for i in range(min(m,n)-1, -1, -1):
        for j in range(i - 1, -1, -1):
            multiplier = system_of_equations[j, i] / system_of_equations[i, i] if system_of_equations[i, i] != 0 else 0
            system_of_equations[j, :] = system_of_equations[j, :] - multiplier * system_of_equations[i, :]
        # I got stuck here for a while; let's discuss
        """
        for j in range(i, -1, -1):
            system_of_equations[j,:] = system_of_equations[j,:] - system_of_equations[j, row_counter] * system_of_equations[i, :]
        row_counter -= 1
        """
    return system_of_equations

if __name__ == "__main__":
    system = np.array(
        [
            [10,23,23,1],
            [3.5,7.9,8,0],
            [42,56,48,5.3]
        ]
    )
    system = guassian_elimination(system)
    for i in range(0, system.shape[0]):
        print(f'x_{i} = {system[i,-1]}')
