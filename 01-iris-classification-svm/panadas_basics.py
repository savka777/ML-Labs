import pandas as pd

data = {'Name' : ['Alice', 'Bob', 'Charlie', 'David'],
        'Age' : [24, 27, 22, 32],}

df = pd.DataFrame(data)

print(df.head())
print(df['Name'])
print(df['Age'])


# Adding a new column
df['City'] = ['New York', 'Los Angeles', 'Chicago', 'Houston']
print(df.head())

# ---------------

# Filtering

df_filtered = df[df['Age'] > 28]

print(df_filtered)


# ---------------

import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x,y)
plt.title("Simple Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
