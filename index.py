import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

### Data Import and display

data = pd.read_csv("salary_dataset.csv")
df = pd.DataFrame(data)
print(df)

### Having taken a look at the table we may assume that the salary is dependent on the education level and experience

x = df[["years_experience", "education_level"]]
y = df["salary"]

### Let's find out the linear relationship between these values

regression = LinearRegression().fit(x,y)
try:
    ### Let's input the experience and education level values and see what the respond will be
    a = float(input("How many years do you work - "))
    b = int(input("What is your education level (1 - High School, 2 - College, 3 - Bachelor's Degree, 4 - Master's degree) - "))

    predicted = regression.predict([[a,b]])
    print(f"Estimated salary of an employee with {a} years of experience and {b} education level: {np.around(predicted[0], decimals=2)}k/annually")
except ValueError:
    print("Error! You must've entered wrong data, try again!")

### Now let's draw the charts to see how salary depends on experience and education level

plt.subplot(1,2,1)
plt.plot(np.sort(df["years_experience"]), np.sort(df["salary"]), color="black")
plt.xlabel("Years of experience")
plt.ylabel("Salary")

plt.subplot(1,2,2)

### The education level plot looks like stairs. Let's make it smooth

edu = np.poly1d(np.polyfit(df["education_level"], df["salary"], 2))
edu_line = np.linspace(0,5,200)

plt.plot(edu_line, edu(edu_line), color="#380082")
plt.xlim(xmin=0, xmax = 6)
plt.xlabel("Education Level")
plt.ylabel("Salary")

plt.suptitle("Salary dependence chart")

plt.show()
