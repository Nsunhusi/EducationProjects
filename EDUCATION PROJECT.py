#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_rows',80)
pd.set_option('display.max_columns',80)
pd.set_option('max_colwidth',300)


# In[3]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.preprocessing import MinMaxScaler


# In[6]:


from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[7]:


data=pd.read_csv('xAPI-Edu-Data.csv')


# In[8]:


data


# In[9]:


data.replace('KW','Kuwait',inplace=True)


# In[10]:


data.replace('KuwaIT','Kuwait',inplace=True)


# In[11]:


data.set_index('gender',inplace=True)


# In[12]:


data.replace(['M', 'L', 'H'], ['SECOND_CLASS', 'THIRD_CLASS', 'FIRST_CLASS'],inplace=True)


# In[13]:


data.replace(['S', 'F'], ['Second_Semester', 'First_Semester'],inplace=True)


# In[14]:


data.drop(['SectionID'],axis=1,inplace=True)


# In[15]:


data.drop(['PlaceofBirth'],axis=1,inplace=True)


# In[16]:


data1=data.reset_index(drop=False)


# In[17]:


data1.replace(['M', 'F'], ['Male', 'Female'],inplace=True)


# In[18]:


data1


# In[19]:


data1['gender'].value_counts()


# In[20]:


data1['raisedhands'].unique()


# In[21]:


def Participation (x):
    if x<=30:
        return 'Not Active'
    elif x<=65:
        return 'Active'
    else:
        return 'Very Active'
data1['Participation_by_hand_raising']=data1['raisedhands'].apply(Participation)


# In[22]:


data1.VisITedResources.unique()


# In[23]:


def Library (x):
    if x<=30:
        return 'Not Frequently'
    elif x<=65:
        return 'Use Moderately'
    else:
        return 'Use Frequently'
data1['Library Usage']=data1['VisITedResources'].apply(Library)


# In[24]:


data1.AnnouncementsView.unique()


# In[25]:


def Notice_Board (x):
    if x<=30:
        return 'Not Regularly'
    elif x<=65:
        return "Read's the Notice Board"
    else:
        return "Always Read's the Notice Board"
data1['Reading of Notice Board']=data1['AnnouncementsView'].apply(Notice_Board)


# In[26]:


data1.Discussion.unique()


# In[27]:


def Participation (x):
    if x<=30:
        return 'Not Active'
    elif x<=65:
        return 'Active'
    else:
        return 'Very Active'
data1['Participation_in_class_discussion']=data1['Discussion'].apply(Participation)


# In[28]:


data1


# In[29]:


data1.StudentAbsenceDays.unique()


# In[30]:


def Attendence (x):
    if x=='Under-7' :
        return 'Not Always Present'
    else:
        return 'Always Present'
data1['Class_Attendence']=data1['StudentAbsenceDays'].apply(Attendence)


# In[31]:


data1


# In[32]:


#sex
fig1=sns.catplot(x='gender',kind='count',data=data1)
fig1
fig1.savefig("output1.png")


# In[33]:


data1.columns


# In[34]:


sns.catplot(x='StageID',hue='gender',kind='count',data=data1);


# In[35]:


def install_opener(opener):
    sns.set(style="darkgrid")
    sns.load_dataset(data1)
sns.pointplot(x="gender", y="Discussion", data=data1);


# In[36]:


sns.pointplot(x="gender", y="Discussion", hue='Class',data=data1);


# In[37]:


sns.pointplot(x="gender", y="VisITedResources", data=data1, capsize=.1)


# In[38]:


sns.pointplot(x="gender", y="VisITedResources", hue='Class',data=data1);


# In[39]:


sns.pointplot(x="gender", y="VisITedResources", hue='Library Usage',data=data1);


# In[40]:


sns.catplot(x="Topic", y="raisedhands",hue="gender", col="Class_Attendence",data=data1, kind="point",aspect=1,height=10);


# In[41]:


data1.Semester.value_counts()


# In[42]:


FS=data1.loc[data1['Semester']=='First_Semester']
FS


# In[43]:


FS.Relation.value_counts()


# In[44]:


sns.catplot(x='Semester',hue='Relation',kind='count',data=FS)


# In[45]:


SS=data1.loc[data1['Semester']=='Second_Semester']
SS


# In[46]:


sns.catplot(x='Semester',hue='Relation',kind='count',data=SS)


# In[47]:


sns.catplot(x='Class',hue='Relation',kind='count',data=data1)


# In[48]:


sns.catplot(x='Class',hue='ParentschoolSatisfaction',kind='count',data=FS)


# In[49]:


sns.catplot(x='Class',hue='ParentschoolSatisfaction',kind='count',data=SS)


# In[50]:


data1.StageID.value_counts()


# In[51]:


LL=data1.loc[data1['StageID']=='lowerlevel']
LL


# In[52]:


LL.NationalITy.value_counts()


# In[53]:


labels = 'Kuwait', 'Palestine', 'Iran', 'lebanon','Tunis','SaudiArabia','Egypt','Syria','Lybia','Iraq','Morocco','USA','Jordan','venzuela'
sizes = [74, 12, 3, 5,2,3,4,3,4,10,1,2,76,1]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','pink','brown','purple','orange','skyblue','blue','red','gray','indigo','green']
explode = (0.1,0.5, 0.5, 0.5,.5,.5,.5,.7,.6,0.5,0.5,.7,0.1,1)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
plt.rcParams['figure.figsize']=(8,10)
plt.axis('equal')
plt.title('COUNTRY_DISTRIBUTION_FOR_LOWERLEVEL_SCHOOL')
plt.show()


# In[54]:


MS=data1.loc[data1['StageID']=='MiddleSchool']
MS


# In[55]:


MS.NationalITy.value_counts()


# In[56]:


labels = 'Kuwait', 'Palestine', 'Iran', 'lebanon','Tunis','SaudiArabia','Egypt','Syria','Lybia','Iraq','Morocco','USA','Jordan','venzuela'
sizes = [84, 16, 2, 12,8,2,5,4,2,12,3,2,96,1]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','pink','brown','purple','orange','skyblue','blue','red','gray','indigo','green']
explode = (0.1,0.5, 0.5, 0.5,.5,.5,.5,.7,.6,0.5,0.5,.7,0.1,1)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
plt.rcParams['figure.figsize']=(8,10)
plt.axis('equal')
plt.title('COUNTRY_DISTRIBUTION_FOR_MIDDLESCHOOL_SCHOOL')
plt.show()


# In[57]:


HS=data1.loc[data1['StageID']=='HighSchool']


# In[58]:


HS.NationalITy.value_counts()


# In[59]:


labels = 'Kuwait', 'Iran','Tunis','SaudiArabia','USA','venzuela'
sizes = [21,1,2,6,2,1]
colors = ['gold', 'lightcoral','pink','purple','gray','indigo']
explode = (0.1,0.5, 0.5, 0.5,.5,.3)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=180)
plt.rcParams['figure.figsize']=(8,10)
plt.axis('equal')
plt.title('COUNTRY_DISTRIBUTION_FOR_HIGHSCHOOL_SCHOOL')
plt.show()


# In[60]:


data1


# In[61]:


data1.drop(['GradeID'],axis=1,inplace=True)


# In[62]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[63]:


data1['Relation']=le.fit_transform(data1['Relation'])


# In[64]:


data1['gender']=le.fit_transform(data1['gender'])


# In[65]:


data1['Semester']=le.fit_transform(data1['Semester'])


# In[66]:


data1['ParentAnsweringSurvey']=le.fit_transform(data1['ParentAnsweringSurvey'])


# In[67]:


data1['ParentschoolSatisfaction']=le.fit_transform(data1['ParentschoolSatisfaction'])


# In[68]:


data1['StudentAbsenceDays']=le.fit_transform(data1['StudentAbsenceDays'])


# In[69]:


data1['Class_Attendence']=le.fit_transform(data1['Class_Attendence'])


# In[70]:


data2=pd.get_dummies(data1,columns=['NationalITy','StageID','Topic','Participation_by_hand_raising','Library Usage','Reading of Notice Board','Participation_in_class_discussion'])
data2


# In[71]:


data2.drop(1)


# In[72]:


data2['Class']=le.fit_transform(data2['Class'])
data2


# In[73]:


X=data2.drop(['Class'],axis=1)


# In[74]:


Y=data2.iloc[:,10:11]


# In[75]:


Y


# In[76]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[77]:


X_train.head()


# In[78]:


Y_train.head()


# In[79]:


from sklearn.preprocessing import StandardScaler
SC_X=StandardScaler()
SC_Y=StandardScaler()


# In[80]:


X_train=SC_X.fit_transform(X_train)


# In[81]:


X_test=SC_X.fit_transform(X_test)


# In[82]:


Y=le.fit_transform(Y)


# In[83]:


#SPLIT DATASET INTO TRAIN AND TEST DATA


# In[84]:


X_train


# In[85]:


X_test


# In[86]:


X_test=pd.DataFrame(X_test)
X_test


# In[87]:


X_train=pd.DataFrame(X_train)
X_train


# In[88]:


from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[89]:


classifier = []
classifier.append(("LogisticReg", LogisticRegression(solver='liblinear',multi_class='ovr')))
classifier.append(("DecisionTree", DecisionTreeClassifier(criterion ='entropy')))
classifier.append(("KNN", KNeighborsClassifier()))
classifier.append(("KernelSVM", SVC(gamma='auto')))
classifier.append(("NaiveBayes", GaussianNB()))
classifier.append(("RandomForest", RandomForestClassifier()))


# In[90]:


seed = 0
results = []
names = []
for name, model in classifier:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[91]:


from sklearn.ensemble import RandomForestClassifier
Classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
Classifier.fit(X_train,Y_train)


# In[92]:


Y_pred=Classifier.predict(X_test)


# In[93]:


Y_pred


# In[94]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[95]:


cm=confusion_matrix(Y_test,Y_pred)


# In[96]:


cm


# In[97]:


accuracyscore=accuracy_score(Y_test, Y_pred)
accuracyscore


# In[ ]:




