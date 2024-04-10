import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

cr_loan = pd.read_csv('/Users/batch/Desktop/python/study/cr_loan.csv')
# Check the structure of the data
print(cr_loan.dtypes)

# Check the first five rows of the data
print(cr_loan.head())

#################
# Check columns with a null value
print(cr_loan.columns[cr_loan.isnull().any()])

# For employment length column, replace with average
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

#With interest rate column, remove them and save the new data without missing data
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index
cr_loan_step1 = cr_loan.drop(indices)
print(cr_loan_step1.columns[cr_loan_step1.isnull().any()])

################
# Create the scatter plot for age and amount
colors = ["blue","red"]
plt.scatter(cr_loan_step1['person_age'], cr_loan_step1['loan_amnt'],
            c = cr_loan_step1['loan_status'],
            cmap = mcolors.ListedColormap(colors),
            alpha=0.5)
plt.colorbar()
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()
plt.clf()

# Drop the individuals older than 100 from the data frame and create a new one
cr_loan_step2 = cr_loan_step1.drop(cr_loan_step1[cr_loan_step1['person_age'] > 100].index)

#Create a scatter plot of age and amount
colors = ["blue","red"]
plt.scatter(cr_loan_step2['person_age'], cr_loan_step2['loan_amnt'],
            c = cr_loan_step2['loan_status'],
            cmap = mcolors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()
plt.clf()

##################

#Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan_step2['loan_status'],cr_loan_step2['person_home_ownership'],
        values=cr_loan_step2['person_emp_length'], aggfunc='max'))

#The max of emp_length is 123 is impossoble -> let's remove!
cr_loan_clean = cr_loan_step2.drop(cr_loan_step2[cr_loan_step2['person_emp_length'] > 60].index)

# Create the new cross table and include maximun and minimum employment length
print(pd.crosstab(cr_loan_clean['loan_status'],cr_loan_clean['person_home_ownership'],
            values=cr_loan_clean['person_emp_length'], aggfunc=['min','max']))

############
with pd.ExcelWriter("clean.xlsx") as writer:
    cr_loan_clean.to_excel(writer, sheet_name="data",index = False)