# Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()

# Required Functions
def Adjust_Family_size(x):
    if x==1:
        return "Single"
    elif x>=2 and x<=3:
        return "Small"
    elif x>=4 and x<=5:
        return  "Medium"
    elif x>=6 and x<=11:
        return "Large"
    else:
        return "Need to find"
def adjust_Tickets(x):
    if x.isdigit():
        return "N"
    else:
        x=x.replace('.','').replace('/','').strip().split(' ')[0]
        return x

def Detect_Outlieris(var):
    global filter
    q1,q3=var.quantile(0.25),var.quantile(0.75)
    iqr=q3-q1
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr
    outlier=var[(var<l_fence)| (var>u_fence)]
    #print("Outliers of ",var.name," : ",outlier.count())
    filter=var.drop(outlier.index, axis = 0)


def train_accuracy(model):
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    train_accuracy = np.round(train_accuracy*100, 2)
    return train_accuracy


#  ::Required Main Code from data processing to model training::
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

data=pd.concat([train,test],sort=False)
#print(data.describe())

#process Data:
#process cabin:
data["Cabin"].fillna(value="X",inplace=True)
data["Cabin"]=data["Cabin"].apply(lambda x: x[0])
#print(data["Cabin"].value_counts())

#process the column "Name"

data["Title"]=data["Name"].str.extract("([a-zA-z]+)\.")
data['Title'].replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace=True)
data['Title'].replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)
data['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
#print(data["Title"].value_counts())
#process no of siblings and parents/Childrens:

data["Family_size"]=data.SibSp+data.Parch+1
data["Family_size"]=data["Family_size"].apply(Adjust_Family_size)
#print(data["Family_size"].value_counts())

#process tickets Colunm:
data["Ticket"]=data["Ticket"].apply(adjust_Tickets)
data["Ticket"]=data["Ticket"].apply(lambda x: x[0])

#print(data["Ticket"].value_counts())

# Detecting Outliers 
# I will use IQR method on column age and Fare
Detect_Outlieris(data["Fare"])
Detect_Outlieris(data["Age"])

# Handling all the missing values 
#print(data.isnull().sum())
# values to be handled are Age, Embarked, Fare and Cabin
# I will handle embarked using mode as this is catagorical variable
#print(data["Embarked"].value_counts())
# S is ocuuring the most 
data["Embarked"].fillna(value="S",inplace=True)
#print(data["Embarked"].value_counts())
# Missing valus in Fare can be handled by median
data["Fare"].fillna(value=data["Fare"].median(),inplace =True)

# Handling missing values in Age 
df = data.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
label = LabelEncoder()
df=df.apply(label.fit_transform)
df['Age'] = data['Age']
df = df.set_index('Age').reset_index()
#print(df.head(5))
plt.figure(figsize=(10,6))
sb.heatmap(df.corr(), cmap ='BrBG',annot = True)
plt.title('Variables correlated with Age')
#plt.show()
# Age have strong relation with pclass and title
data["Age"]=data.groupby(["Pclass","Title"])["Age"].transform(lambda x: x.fillna(x.median()))
#print(data.isnull().sum())

# Now i will handle contineous values in some columns like:
# Age and  Fare
#I will overcome thier continous will use by making them labels
#I will label them in accordance to thier ranges
# Age:
label_names = ['infant', 'child', 'teenager','young_adult', 'adult', 'aged']
cut_points = [0,5,12,18,35,60,81]
data['New_age'] = pd.cut(data['Age'], cut_points, labels = label_names)
#print(data[['Age', 'New_age']].head(2))
# Fare:
groups = ['low','medium','high','very_high']
cut_points = [-1, 130, 260, 390, 520]
data['New_fare'] = pd.cut(data["Fare"], cut_points, labels = groups)
#print(data[['Fare', 'New_fare']].head(2) )  

# Correcting data types.
# I need to analyze all the data types 
#print(data.dtypes)
data['Pclass'] = data['Pclass'].astype('category')
data['Sex'] = data['Sex'].astype('category')
data['Embarked'] = data['Embarked'].astype('category')
data['Cabin'] = data['Cabin'].astype('category')
data['Title'] = data['Title'].astype('category')
data['Family_size'] = data['Family_size'].astype('category')
data['Ticket'] = data['Ticket'].astype('category')
data['Survived'] = data['Survived'].dropna().astype('int')
#print(data.dtypes)

# Time to drop some columns 
# from the above handling i think i should drop 
# Age, sibSB, Name,parch and Fare
data.drop(columns = ['Name', 'Age','SibSp', 'Parch','Fare'], inplace = True, axis = 1)
#print(data.columns)  

# Applying on hot encoding on catogorical data
data=pd.get_dummies(data,drop_first=True)
#print(data.head(7))

# appling Machine learning algorithms
# algorithms i will apply are:
# 1. Logistic regression  2.Knn  3. SVM
# 4. XGboost 5. Random Forest
train = data.iloc[:891, :]
test  = data.iloc[891:, :] 
train = train.drop(columns = ['PassengerId'], axis = 1)
test = test.drop(columns = ['Survived'], axis = 1)
X_train = train.drop(columns = ['Survived'], axis = 1) 
y_train = train['Survived']
X_test  = test.drop("PassengerId", axis = 1).copy()


#1.Logistic Regression
lr = LogisticRegression()
#2.KNN
knn = KNeighborsClassifier()
#3.Decision Tree Classifier
dt = DecisionTreeClassifier(random_state = 40)
#4.Random Forest Classifier
rf = RandomForestClassifier(random_state = 40, n_estimators = 100)
#5.Support Vector Machines
svc = SVC(gamma = 'auto')
#6. XGBoost 
xgb = XGBClassifier(n_job = -1, random_state = 40)

trained_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), train_accuracy(knn), train_accuracy(dt), train_accuracy(rf), train_accuracy(svc), train_accuracy(xgb)]})
trained_accuracy.index = ['LR', 'KNN','DT', 'RF', 'SVC', 'XGB']
trained_accuracy = trained_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)
print(trained_accuracy)

# Dt was a great fit i think
# because it gave me 89.79% accuracy (highest of all)

#https://www.kaggle.com/code/vikassingh1996/titanic-classification-comprehensive-modeling/notebook#3.5-Outliers-Detection-


