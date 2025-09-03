import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

#Definition of necessary parameters:
#----------------------------------------------------------
N           = 20 #Choose the random numbers to be generated.
experiments = 10000  #Choose the number of experiments. 
#-----------------
a_hat_values           =  []
b_hat_values           =  []
chi_square_mins        =  []
p_values               =  []
a_true                 =  5.0
b_true                 = -0.5

# Linear function:
def linear_model(x, a, b):
    return a + b * x

# Quadratic function:
def quadratic_model(x, a, b, c):
    return a + b * x + c * x**2

#Chi-square function:
def chi_square(model, y, e):
    chi_square = np.sum((y - model) ** 2 / e ** 2)
    return chi_square

for _ in range(experiments):
    # Important parameters:
    N = 10
    x = np.random.uniform(0, 10, N)
    μ = -0.5 * x + 5  # Mean 
    e = 0.1           # Variance
    y = np.random.normal(μ, e, N)

    
    if experiments == 1:
        params, cov_matrix = curve_fit(linear_model, x, y) 
        params_quad, cov_matrix_quad = curve_fit(quadratic_model, x, y) 

        # The parameters from linear model:
        print("Coefficients from linear model: a =", params[0], "b =", params[1])

        # The parameters from quadratic model:
        print("Coefficients from quadratic model: a =", params_quad[0],
              "b =", params_quad[1], "c =", params_quad[2])

        # Plot of observations from random numbers generation, the linear model, and the quadratic model:
        x_model = np.linspace(0, 10, 1000)
        
        plt.errorbar(x, y, yerr=e, fmt='o', color='cyan', alpha=1.0, label=f"Observations (N = {N})")
        plt.plot(x_model, linear_model(x_model, params[0], params[1]), color='black', label="Linear model", linewidth=2.5)
        plt.plot(x_model, quadratic_model(x_model, params_quad[0], params_quad[1], params_quad[2]), color='red', label="Quadratic model", linewidth=2.0)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Observations, Linear Model & Quadratic Model")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Covariance matrix for the linear model:
        print("Covariance matrix for linear model y = b * x + a:")
        print(cov_matrix)

        # Covariance matrix for the quadratic model:
        print("Covariance matrix for quadratic model y = c * x ** 2 + b * x + a:")
        print(cov_matrix_quad)
    
    
    params, _ = curve_fit(linear_model, x, y) 
    a_hat_values.append(params[0])
    b_hat_values.append(params[1])

    #Calculation of chi-squares for the number of experiments:
    chi_squares = chi_square(linear_model(x, params[0], params[1]), y, e)
    chi_square_mins.append(chi_squares)

#All the p-values calculation:
for i in chi_square_mins: 
    count = np.sum(np.array(chi_square_mins) >= i)
    p_values.append(count / experiments) 

#p-value:
mean_chi_square_min = np.mean(chi_square_mins)
count = np.sum(np.array(chi_square_mins) >= mean_chi_square_min)
p_value = count / experiments

#Mean and standard deviation of p-value:
mean_p_value        = np.mean(p_values)
std_dev_p_value     = np.std(p_values)

#Standard deviation of a_hat, b_hat:
std_a_hat = np.std(a_hat_values)
std_b_hat = np.std(b_hat_values)

#Necessary quantities: A = (a_hat-a_true) / sigma_a, B = (b_hat-b_true )/ sigma_b:
A = (np.array(a_hat_values) - a_true) /np.std(a_hat_values)
B = (np.array(a_hat_values) - a_true) /np.std(a_hat_values)

#Parameter a_hat histogram:
plt.hist(a_hat_values, bins = 30, density = True, color = 'cyan', edgecolor = 'black', alpha = 0.8)
plt.xlabel("$\\hat{a}$", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.title("Parameter $\\hat{a}$")
plt.grid(True)
plt.show()

#Parameter b_hat histogram:
plt.hist(b_hat_values, bins = 30, density = True, color = 'cyan', edgecolor = 'black', alpha = 0.8)
plt.xlabel("$\\hat{b}$", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.title("Parameter $\\hat{b}$")
plt.grid(True)
plt.show()

#Parameter A histogram:
plt.hist(A, bins = 30, density = True, color = 'cyan', edgecolor = 'black', alpha = 0.8)
plt.xlabel("$(\\hat{a} - a_{true}) / \\sigma_{\\hat{a}}$", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.title("Parameter $(\\hat{a} - a_{true}) / \\sigma_{\\hat{a}}$")
plt.grid(True)
plt.show()

#Parameter B histogram:
plt.hist(B, bins = 30, density = True, color = 'cyan', edgecolor = 'black', alpha = 0.8)
plt.xlabel("$(\\hat{b} - b_{true}) / \\sigma_{\\hat{b}}$", fontsize = 10)
plt.ylabel("Density", fontsize = 10)
plt.title("Parameter $(\\hat{b} - b_{true}) / \\sigma_{\\hat{b}}$")
plt.grid(True)
plt.show()

#Mins of chi-squares histogram:
plt.hist(chi_square_mins, bins = 30, density = True, color = "grey", edgecolor = "black", alpha = 0.5)
plt.xlabel('Chi Square values', fontsize = 10)
plt.ylabel('Density', fontsize = 10)
plt.title('Chi squares Distribution')
plt.grid(True)
plt.show()

#p-values histogram:
plt.hist(p_values, bins = 30, density=True, color='grey', edgecolor='black', alpha=0.7)
plt.xlabel('p-values', fontsize = 10)
plt.ylabel('Density', fontsize = 10)
plt.title('p-values Distribution')
plt.grid(True)
plt.show()

#Results: 
print("The chi square min has a mean value equal to", np.mean(chi_square_mins), 
       "and standard deviation: ", np.std(chi_square_mins))
print("The p-value is:", p_value)
print("The mean of p-value is:", mean_p_value)
print("The a_hat has mean ", np.mean(a_hat_values), "and standard deviation: ", std_a_hat)
print("The b_hat has mean ", np.mean(b_hat_values), "and standard deviation: ", std_b_hat)
print("(a - a_true) / std_dev_a: Mean =", np.mean(A))
print("(b - b_true) / std_dev_b: Mean =", np.mean(B))

print(np.array(a_hat_values))
print(np.array(a_true))