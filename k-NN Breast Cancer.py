#-  MASOUD RAFIEE
# - INTRO TO AI - FALL 2024
# - MACHINE LEARNING: k-NN ALGORITHM ON BREAST CANCER DATASET
##################################################################
#load breast cancer dataset
from sklearn.datasets import load_breast_cancer
# numerical ops and handling array
import numpy as np
# For creating heatmaps
import seaborn as sns
#manip dataframes (or data manip and analysis)
import pandas as pd
#creating static interactive / or animated visulaizations
import matplotlib.pyplot as plt #plotting for 2D - 3D
#for 3D scatter plot and coloring (a toolkit for matplot)
#for training and testing the model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#1. DATASET OVERVIEW
data = load_breast_cancer() #loading the breast cancer dataset to data (similar to dictionary)
df=pd.DataFrame(data.data,columns=data.feature_names) #putting features name and data (2-D) of breast cancer to dataframe of pandas
df['target']=data.target #labels of 0 or 1s for being cancerous (benign for 0, malignant 1)
print(df.head().to_string()) #render df.head to a consule tabular output


#2. 2-D SCATTER PLOT
rand_select = np.random.choice(df.columns[:-1], size=2, replace=False) #using numpy for selecting 2 non-duplicated columns
x_column , y_column = rand_select #unpacking the two names for easier access
plt.figure(figsize=(6,5))#creating a 6x5 figure (inches)
plt.scatter(df[x_column],df[y_column],c=df['target'], cmap='RdBu', alpha=0.7,edgecolors='k') #creating scatter plot of x and y with transpareny of .5 and edge color of data point black
plt.title(f'2D Scatter Plot: {x_column}, vs {y_column}', fontsize=13)#adding a title
plt.xlabel(x_column, fontsize=12)#labeling x axis
plt.ylabel(y_column, fontsize=12)#labeling y axis as y
plt.colorbar(label='(0 = Benign, 1 = Malignant)')
plt.show() #displaying the scatter plot


#3. 3-D COLORED SCATTER PLOT
rand_select_3D = np.random.choice(df.columns[:-1],size=3, replace=False) #col-1 because the last col is target
x_column1, y_column1, z_column1 = rand_select_3D
figure = plt.figure(figsize=(10,8))
ax= figure.add_subplot(1,1,1,projection='3d') #make a 3d subplot with #of rows, #of cols, #of plot in that 1x1 grid
scatter = ax.scatter(df[x_column1],df[y_column1],df[z_column1],c=df['target'], cmap='RdBu', s=70, alpha=0.7, edgecolors='k')
ax.set_title(f'3D Scatter Plot: {x_column1}, vs , {y_column1}, vs , {z_column1}', fontsize=14)
#setting labels to each randomly chosen column
ax.set_xlabel(x_column1, fontsize=12)
ax.set_ylabel(y_column1, fontsize=12)
ax.set_zlabel(z_column1, fontsize=12)
cbar=plt.colorbar(scatter,ax=ax, shrink=0.8, aspect =10)
cbar.set_label('(0 = Benign, 1 = Malignant)', fontsize=13)
plt.show()


#4. Model Training and Evaluation:
X=df.drop(columns=['target']) #features (not the target)
y=df['target'] #target labels
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8, random_state=42)
kNN=KNeighborsClassifier(n_neighbors=5) #k=5
kNN.fit(X_train, y_train)#to train the data
y_pred = kNN.predict(X_test) #predict on the test set
accuracy=accuracy_score(y_test,y_pred) #hoew many pred correct compared to true labels
accuracy=accuracy*100 #to make it as percentage to look better
print(f"Accuracy of k-NN (k=5): {accuracy: .0f} % ")


#5. Exploring Different Values of k and different split size:
k_values=range(1,11) #from 1 to 10
test_sizes=[0.1,0.2,0.4]#test sizes proportion, 10% and so on
accuracy_matrix = np.zeros((len(k_values),len(test_sizes)))
for i, test_size in enumerate(test_sizes):
    for j, k in enumerate (k_values):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        kNN=KNeighborsClassifier(n_neighbors=k) #initialize k-nn classifier
        kNN.fit(X_train,y_train) #train the model
        y_pred = kNN.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        accuracy_matrix[j, i] = accuracy  #store the accuracy for the current k value (row) and test size (column)
plt.figure(figsize=(8,6))
sns.heatmap(
    accuracy_matrix,#requied data to visualzie heatmap
    annot=True,#show accuracy values in the cells
    fmt=".2f", #format values to 2 decimal places
    xticklabels=[f"{int(s*100)}%" for s in test_sizes],  #converting test size to -> % (x-axis)
    yticklabels=k_values, #labeling rows with k values (y-axis)
    cmap="coolwarm", #blue red color
    cbar_kws={'label':'Accuracy'} #Add a label to the collor bar
    )
plt.xlabel("Test Size (%)")#label for the x-axis
plt.ylabel("k Value")#label for y axis
plt.title("k-NN Performance with Different k Values and Train/Test splits")
plt.show()



