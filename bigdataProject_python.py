#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
dt_dataset = pd.read_csv("query_result.csv")


# In[12]:


dt_dataset.isnull().sum()


# In[13]:


unknown_cols = []
for col in dt_dataset.columns:
    unknown_count = dt_dataset[col].apply(lambda x: x.lower() 
                                          if type(x) == str 
                                          else x).value_counts().get('unknown', 0)
    if unknown_count > 0:
        unknown_cols.append(col)

# Print the list of columns that contain unknown string values
print("The following columns contain unknown string values:")
print(unknown_cols)


# In[14]:


dt_dataset.head()


# In[33]:


from sklearn.feature_extraction import FeatureHasher
import pandas as pd

cat_cols = [ 'model', 'driverid', 'event']
uc=['city','state', 'tdate']

# Create the FeatureHasher object
n_hash_bits = 8
hasher = FeatureHasher(n_features=2**n_hash_bits, input_type='string')
hashed = hasher.transform(dt_dataset[cat_cols].astype(str).values)
hashed_df = pd.DataFrame(hashed.todense(), columns=[f'hash_{i}' for i in range(2**n_hash_bits)])
data_encoded = pd.concat([dt_dataset, hashed_df], axis=1)
data_encoded = data_encoded.drop(cat_cols, axis=1)
data_encoded = data_encoded.drop(uc, axis=1)
# standardizeing the data
cols_to_standardize = ["velocity", "events", "totmiles", "riskfactor", 'miles', 'gas' ]
scaler = StandardScaler()

# Fit the scaler to the selected columns
scaler.fit(data_encoded[cols_to_standardize])

# Transform the selected columns using the fitted scaler
data_encoded[cols_to_standardize] = scaler.transform(data_encoded[cols_to_standardize])


# In[30]:


data_encoded.head()


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


# standardizeing the data
cols_to_standardize = ["velocity", "events", "totmiles", "riskfactor", 'miles', 'gas' ]
scaler = StandardScaler()

# Fit the scaler to the selected columns
scaler.fit(dt_dataset[cols_to_standardize])

# Transform the selected columns using the fitted scaler
dt_dataset[cols_to_standardize] = scaler.transform(dt_dataset[cols_to_standardize])


# In[20]:


dt_dataset.head()


# In[44]:


import statsmodels.api as sm
import pandas as pd

# create the predictor and response variables
x=dt_dataset[['velocity','events','totmiles']]
y = dt_dataset['riskfactor']

# add a constant column to the predictor variables
x = sm.add_constant(x)

# create a linear regression model and fit it to the data
model = sm.OLS(y, x).fit()

# print the model summary
print(model.summary())


# In[34]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Iterate over values of k from 2 to 7
for k in range(2, 8):
    # Initialize KMeans object with k clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    # Fit the KMeans object to the data
    kmeans.fit(data_encoded)
    # Compute the silhouette score for this clustering
    score = silhouette_score(data_encoded, kmeans.labels_)
    # Print the silhouette score for this value of k
    print(f"For k={k}, the silhouette score is {score:.3f}")


# In[63]:


# Perform k-means clustering with the best k value (let's say k=3)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_encoded)

# Add a new column to the data frame with the cluster labels
data_encoded['Cluster'] = kmeans.labels_

# Print out the whole data along with the cluster labels assigned for each row
print(data_encoded)


# In[64]:


cluster_means = data_encoded.groupby('Cluster').mean()
print(cluster_means)


# In[65]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
scatter = ax.scatter(data_encoded['velocity'], data_encoded['riskfactor'], c=data_encoded['Cluster'])
legend = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Clusters")
ax.add_artist(legend)
plt.show()


# In[66]:


import seaborn as sns

sns.pairplot(data_encoded, vars=['events','velocity','totmiles', 'riskfactor'], hue='Cluster', palette='Dark2')
plt.show()


# In[46]:


pip install bk_clustering


# In[47]:


pip install PyHive


# In[49]:


pip install thrift


# In[54]:


pip install thrift_sasl


# In[52]:


pip install sasl


# In[121]:


from pyhive import hive
import pandas as pd


# In[122]:


conn = hive.Connection(host="10.182.131.237", port=10000, username="training")


# In[123]:


cursor = conn.cursor()
cursor.execute("SELECT * FROM model")
result = cursor.fetchall()


# In[124]:


conn.close()


# In[178]:


df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])


# In[179]:


df.head()


# In[180]:


# define the range for binary conversion
bin_range = [1, 7, 32]

# convert the values in the column to binary based on the range
df['riskfactor_1'] = pd.cut(df['model.riskfactor'], bins=bin_range, labels=[0, 1])


# In[181]:


cols_to_standardize = ["model.velocity", "model.events", "model.totmiles", "model.riskfactor"]
scaler = StandardScaler()

# Fit the scaler to the selected columns
scaler.fit(df[cols_to_standardize])

# Transform the selected columns using the fitted scaler
df[cols_to_standardize] = scaler.transform(df[cols_to_standardize])


# In[182]:


df.head()


# In[158]:


import pandas as pd

# Assume you have a dataframe called 'df' with a column called 'driverID'
# that contains the categorical variable you want to encode

# Convert the categorical column to a pandas category type
df['model.driverid'] = df['model.driverid'].astype('category')

# Create a new dataframe with one column for each unique category in driverID
one_hot = pd.get_dummies(df['model.driverid'])

# Add the new columns to the original dataframe
df = pd.concat([df, one_hot], axis=1)

# Drop the original driverID column, since it has been replaced with the one-hot encoded columns
df.drop('model.driverid', axis=1, inplace=True)


# In[159]:


df.head()


# In[160]:


import statsmodels.api as sm
import pandas as pd

# create the predictor and response variables
x=df[['model.velocity','model.events','model.totmiles']]
y = df['model.riskfactor']

# add a constant column to the predictor variables
x = sm.add_constant(x)

# create a linear regression model and fit it to the data
model = sm.OLS(y, x).fit()

# print the model summary
print(model.summary())


# In[161]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Iterate over values of k from 2 to 7
for k in range(2, 10):
    # Initialize KMeans object with k clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    # Fit the KMeans object to the data
    kmeans.fit(df)
    # Compute the silhouette score for this clustering
    score = silhouette_score(df, kmeans.labels_)
    # Print the silhouette score for this value of k
    print(f"For k={k}, the silhouette score is {score:.3f}")


# In[162]:


# Perform k-means clustering with the best k value (let's say k=3)
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

# Add a new column to the data frame with the cluster labels
df['Cluster'] = kmeans.labels_

# Print out the whole data along with the cluster labels assigned for each row
print(df)


# In[163]:


cluster_means = df.groupby('Cluster').mean()
print(cluster_means)


# In[165]:


import seaborn as sns

sns.pairplot(df, vars=['model.events','model.velocity','model.totmiles', 'model.riskfactor','A50'], hue='Cluster', palette='Dark2')
plt.show()


# In[101]:


cursor = conn.cursor()
cursor.execute("SELECT * FROM trucks")
result = cursor.fetchall()


# In[103]:


df1 = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])


# In[104]:


df1.head()


# In[200]:


data1=df.drop('model.driverid', axis=1)


# In[201]:


data2=data1.drop('model.riskfactor', axis=1)


# In[202]:


data=data2.drop('model.events', axis=1)


# In[203]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('riskfactor_1', axis=1), data['riskfactor_1'], test_size=0.3, random_state=42)

# Create a Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing dat
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[199]:


data.head()


# In[207]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Iterate over values of k from 2 to 7
for k in range(2, 10):
    # Initialize KMeans object with k clusters
    kmeans = KMeans(n_clusters=k, random_state=0)
    # Fit the KMeans object to the data
    kmeans.fit(data2)
    # Compute the silhouette score for this clustering
    score = silhouette_score(data2, kmeans.labels_)
    # Print the silhouette score for this value of k
    print(f"For k={k}, the silhouette score is {score:.3f}")


# In[212]:


# Perform k-means clustering with the best k value (let's say k=3)
kmeans = KMeans(n_clusters=8, random_state=0).fit(data2)

# Add a new column to the data frame with the cluster labels
data2['Cluster'] = kmeans.labels_

# Print out the whole data along with the cluster labels assigned for each row
print(data2)


# In[213]:


cluster_means = data2.groupby('Cluster').mean()
print(cluster_means)


# In[214]:


import seaborn as sns

sns.pairplot(data2, vars=['model.events','model.velocity','model.totmiles'], hue='Cluster', palette='Dark2')
plt.show()


# In[ ]:




