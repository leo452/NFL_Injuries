# Import dependencies
import pandas as pd
import numpy as np


#Carga de los datos al sistema
df = pd.read_csv('data/InjuryRecord.csv')
print("llego")
df2 = pd.read_csv('data/PlayList.csv')
print("llego1")
df3 = pd.read_csv('data/PlayerTrackData.csv')
print("llego2")

Injury_Rec = df
#Injury_Rec.head()

PlayList = df2
#PlayList.head()

Tracks = df3

inj_detailed_merged = Injury_Rec.merge(PlayList)

ds = inj_detailed_merged.groupby(['RosterPosition','BodyPart']).count() \
.unstack('BodyPart')['PlayerKey']\
.T.apply(lambda x: x / x.sum())\
.sort_values('BodyPart').T.sort_values('Ankle', ascending=False)

df = inj_detailed_merged.groupby(['PlayType','BodyPart']).count() \
.unstack('BodyPart')['PlayerKey']\
.T.apply(lambda x: x / x.sum())\
.sort_values('BodyPart').T.sort_values('Ankle', ascending=False)\

df = inj_detailed_merged.groupby(['RosterPosition','Surface']).count() \
.unstack('Surface')['PlayerKey']\
.T.apply(lambda x: x / x.sum())\
.sort_values('Surface').T.sort_values('Natural', ascending=False)

Injury_Rec

PlayList

Tracks

Jugadas_lesiones= PlayList.merge(Injury_Rec, how= "inner")
Jugadas_lesiones2= Jugadas_lesiones.merge(PlayList, how="outer")

Jugadas_lesiones2

Jugadas_lesiones2['BodyPart']=Jugadas_lesiones2.BodyPart.fillna("None")

Jugadas_lesiones2=Jugadas_lesiones2.drop("Surface",axis=1)

Jugadas_lesiones2['DM_M1']=Jugadas_lesiones2.DM_M1.fillna(0)
Jugadas_lesiones2['DM_M7']=Jugadas_lesiones2.DM_M7.fillna(0)
Jugadas_lesiones2['DM_M28']=Jugadas_lesiones2.DM_M28.fillna(0)
Jugadas_lesiones2['DM_M42']=Jugadas_lesiones2.DM_M42.fillna(0)

for ind in Jugadas_lesiones2.index:
    if Jugadas_lesiones2["DM_M1"][ind]==1 and Jugadas_lesiones2["DM_M7"][ind]==0:
        continue
    elif Jugadas_lesiones2["DM_M1"][ind]==1 and Jugadas_lesiones2["DM_M7"][ind]==1 and Jugadas_lesiones2["DM_M28"][ind]==0:
        Jugadas_lesiones2["DM_M1"][ind]=0
    elif Jugadas_lesiones2["DM_M1"][ind]==1 and Jugadas_lesiones2["DM_M7"][ind]==1 and Jugadas_lesiones2["DM_M28"][ind]==1 and Jugadas_lesiones2["DM_M42"][ind]==0:
        Jugadas_lesiones2["DM_M1"][ind]=0
        Jugadas_lesiones2["DM_M7"][ind]=0
    elif Jugadas_lesiones2["DM_M1"][ind]==1 and Jugadas_lesiones2["DM_M7"][ind]==1 and Jugadas_lesiones2["DM_M28"][ind]==1 and Jugadas_lesiones2["DM_M42"][ind]==1:
        Jugadas_lesiones2["DM_M1"][ind]=0
        Jugadas_lesiones2["DM_M7"][ind]=0
        Jugadas_lesiones2["DM_M28"][ind]=0

data_set=Jugadas_lesiones2.copy()
data_set=data_set.drop("PlayerKey",axis=1)
data_set=data_set.drop("PlayKey",axis=1)
data_set=data_set.drop("GameID",axis=1)
data_set=data_set.drop("PlayerGamePlay",axis=1)

data_set["Lesion"]=0

for ind in data_set.index:
    if data_set["DM_M1"][ind]==1 or data_set["DM_M7"][ind]==1 or data_set["DM_M28"][ind]==1 or data_set["DM_M42"][ind]==1:
        data_set["Lesion"][ind]=1

jugadas_lesiones3 = Jugadas_lesiones2.copy()
jugadas_lesiones3

jugadas_lesiones3["Total_DM"]=0
for ind in jugadas_lesiones3.index:
    if jugadas_lesiones3["DM_M1"][ind]==1:
        jugadas_lesiones3["Total_DM"][ind]=1
    elif jugadas_lesiones3["DM_M7"][ind]==1:
        jugadas_lesiones3["Total_DM"][ind]=7
    elif jugadas_lesiones3["DM_M28"][ind]==1:
        jugadas_lesiones3["Total_DM"][ind]=28
    elif jugadas_lesiones3["DM_M42"][ind]==1:
        jugadas_lesiones3["Total_DM"][ind]=42

jugadas_lesiones3.head()

jugadas_lesiones3.Total_DM.unique()

jugadas_lesiones3=jugadas_lesiones3.drop("DM_M1",axis=1)
jugadas_lesiones3=jugadas_lesiones3.drop("DM_M7",axis=1)
jugadas_lesiones3=jugadas_lesiones3.drop("DM_M28",axis=1)
jugadas_lesiones3=jugadas_lesiones3.drop("DM_M42",axis=1)

data_set=data_set.drop("DM_M1",axis=1)
data_set=data_set.drop("DM_M7",axis=1)
data_set=data_set.drop("DM_M28",axis=1)
data_set=data_set.drop("DM_M42",axis=1)

data_set

from sklearn import preprocessing
data_set2=data_set.copy()
data_set2=data_set2.drop("BodyPart",axis=1)
le = preprocessing.LabelEncoder()
data_set2.RosterPosition = le.fit_transform(data_set2.RosterPosition)
data_set2.StadiumType = le.fit_transform(data_set2.StadiumType)
data_set2.FieldType = le.fit_transform(data_set2.FieldType)
data_set2.Weather = le.fit_transform(data_set2.Weather)
data_set2.PlayType = le.fit_transform(data_set2.PlayType)
data_set2.Position = le.fit_transform(data_set2.Position)
data_set2.PositionGroup = le.fit_transform(data_set2.PositionGroup)
#data_set2.BodyPart = le.fit_transform(data_set2.BodyPart)

data_set2

X= data_set2.drop("Lesion",axis=1)
y= data_set2["Lesion"]


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

from sklearn.model_selection import GridSearchCV

tree_para = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8,9]}

clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

clf.best_estimator_

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

len(clf.feature_importances_)

# Save your model
#from sklearn.externals import joblib
import joblib
joblib.dump(clf, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
clf = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")