#!/usr/bin/env python
# coding: utf-8

# ### Projet de modélisation de détresse financière
# 
# Dans ce kernel, nous allons nous intéressez au problème de stress financier auquel sont exposé les entreprise
# 
# Pour ce faire, nous allons analyser un ensemble de données issues d'une ancienne compétition kaggle. L'ensemble des données traite de la prévision de détresse financière pour un échantillon d’entreprises. Mais avons, commencer par définir la notion de détressae financière afin de mieux cerner l'enjeux métier.
# 
# ### Fiancial stress analysis
# 
# Les crises financières causent des ravages économiques, sociaux et politiques. Les politiques macroprudentielles gagnent du terrain, mais sont encore très peu étudiées par rapport à la politique monétaire et à la politique budgétaire. Nous utilisons le cadre général des prédictions séquentielles également appelé apprentissage automatique en ligne pour prévoir les crises hors échantillon.
# 
# Le risque systémique financier est une question importante dans les systèmes économiques et financiers. En essayant de détecter et de réagir au risque systémique avec des quantités croissantes de données produites sur les marchés financiers et les systèmes, beaucoup de chercheurs ont de plus en plus utilisé des méthodes d’apprentissage automatique. Les méthodes d’apprentissage automatique étudient les mécanismes d’éclosion et de contagion du risque systémique dans le réseau financier et améliorent la réglementation actuelle du marché financier et de l’industrie. 
# 
# Ainsi dans ce kernel, Dans cet article, nous nous basons sur les recherches et méthodologies existantes sur l’évaluation et la mesure du risque systémique financier combinées aux technologies d’apprentissage automatique, y compris l’analyse du Big Data, l’analyse du réseau et l’analyse des sentiments
# 
# 
# Maintenant, intéressons nous aux données Kaggle.

# #### Description des données
# 
# Company: La société représente des entreprises échantillonnées.
# 
# Time : Le temps montre différentes périodes de temps à laquelle appartiennent les données. La durée des séries horaires varie entre 1 et 14 pour chaque entreprise.
# 
# Fiancial Ditress: La variable cible est indiquée par «détresse financière» si elle sera supérieure à -0,50, l’entreprise doit être considérée comme saine (0). Dans le cas contraire, il serait considéré comme financièrement en difficulté (1).
# 
# Le rete des colonnes : Les caractéristiques indiquées par x1 à x83, sont quelques caractéristiques financières et non financières des sociétés échantillonnées. Ces caractéristiques appartiennent à la période précédente, qui devrait être utilisée pour prédire si l’entreprise sera financièrement en difficulté ou non (classification). Caractéristique x80 est variable catégorique.

# In[1]:


#Import de librairies nécessaire
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# Sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit


# In[2]:


df=pd.read_csv('financial stress.csv')
df.head(20)


# Par exemple, l’entreprise 1 est en difficulté financière à l’époque 4, mais l’entreprise 2 est toujours en bonne santé au moment 14.

# In[3]:


#Vérification de la présence de donnée manquantes

print("Total missing values:", df.isna().sum().sum())


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


## Distrees financières selon la parcelle d'heures de travail

df=df.rename(columns={'Financial Distress':'Financial Distress'})
df.head(5)


# La description de Kaggle nous indique que si le nombre dans la colonne "détresse financière" est inférieur à -0,5, la société doit être considérée comme en difficulté.
# 
# Nous pouvons imaginer qu'il s'agit d'une sorte de ratio financier - ratio des revenus sur le capital ou autre.

# ### Visualisation du détresse financier en fonction du temps

# In[8]:


plt.scatter(df['Time'],df['Financial Distress'],color='red')
plt.xlabel('Time')
plt.ylabel('Financial Distress')


# ### Exploration des données
# 
# 1- Cherchons le nombre d'entreprises uniques.
# 
# 2- Vérifions combien de ces entreprises ont atteint un état de détresse (136 selon la description de Kaggle).
# 
# 3- Cherchon une liste des noms de caractéristiques.

# In[9]:


total_n = len(df.groupby('Company')['Company'].nunique())
print(total_n)


# On identifie alors 422 entreprise uniques

# In[10]:


distress_companies = df[df['Financial Distress'] < -0.5]
u_distress = distress_companies['Company'].unique()
print(u_distress.shape)


# On identifie effectivement 136 entreprises qui ont atteint un état de détresse financière. Cela indique que cet ensemble de données est déséquilibré et biaisées car on y compte 136 entreprises en difficulté financière contre 286 entreprises en bonne santé, c’est-à-dire que 136 entreprises de l’année sont en difficulté financière alors que 3546 entreprises en année sont en bonne santé. De ce fait, nous pourrions utilier le f-score comme critère d’évaluation du rendement.

# In[11]:


# Obtenons une liste de noms de caractéristiques des entreprises.

feature_names = list(df.columns.values)[3:] 
print(feature_names)


# ### Type d'apprentissage à appliquer
# 
# Au vu de ces données, on peut faire avancer les hypothèses suivantes: 
#     
#    1- Ces données peuvent être considérées comme un problème de classification.
# 
#    2- Ces données pourraient également être considérées comme un problème de régression, puis le résultat sera converti en classification.
# 
#    3- Ces données pourraient être considérées comme une classification multivarié des séries invariables.
# 
# Les questions à se poser sont les suivantes: 
#     
#    a- Quelles sont les caractéristiques les plus révélatrices de la détresse financière?
# 
#    b- Quels types de modèles d’apprentissage automatique sont les plus performants sur cet ensemble de données ?

# ### Choix du temps idéal pour séparer les données d'apprentissage et de test
# 
# Pour mener à bien cette analyse, il convient de scinder les données en apprentissage, validation et test. Ainsi, afin de choisir une bonne date pour séparer le train et les essais, nous devrions idéalement choisir une date qui permette à la plupart des entités d'apparaître à la fois dans les données du train et des essais.
# 
# Malheureusement, toutes les compagnies ne vivent pas pendant la même durée, donc si nous choisissons une date trop précoce ou trop tardive, nous risquons de retirer de nombreuses compagnies de la série de tests.
# 
# Générons un histogramme des comptages pour chaque période afin de pouvoir choisir un endroit raisonnable pour la suppression.

# In[12]:


df.hist(column=['Time'], bins=14)


# Nous constatons un léger déclin, puis une hausse dans l'histogramme autour de la période 10.
# 
# Les baisses impliquent qu'une entreprise disparaît de l'ensemble de données, donc si nous fixons notre réduction autour de t=10, nous devrions encore obtenir un nombre décent de cas de détresse dans les données de formation.

# In[13]:


print(df.groupby(['Company'])['Time'].agg('min'))

    # Nous pouvons voir que la plupart des entreprises commencent à la période 1, 
    # mais il y en a qui commencent leur vie beaucoup plus tard.


# ### Vérifions maintenant si la détresse se produit-elle de manière uniforme dans le temps dans ces entreprises

# In[14]:


distress_companies.hist(column=['Time'], bins=14)


# On constate que la fréquence de la détresse ne semble certainement pas être uniforme dans le temps.
# 
# Cela indique qu'il peut être malavisé d'obtenir des validations ou des jeux de tests en choisissant simplement certaines entreprises, car nous ne pouvons pas supposer que les différentes entreprises sont indépendantes. L'horodatage lui-même peut être un signal utile (c'est-à-dire si une certaine période représente un état de déclin macroéconomique pour une certaine industrie, ou l'économie dans son ensemble).

# In[15]:


f80 = list(df.groupby('Company')['x80'].agg('mean'))
f80 = [int(c) for c in f80]


# ### Validation croisée: pour séparer les données en apprentissage, validation, test

# In[16]:


# Generation de la base d'apprentissage et la bae test par validation croisée

datadict = {}
distress_dict = {}

for i in range (1, total_n+1):
    datadict[i] = {}
    distress_dict[i] = {}

print("Populating dictionary...")
for idx, row in df.iterrows():
    company = row['Company']
    time = int(row['Time'])
    
    datadict[company][time] = {}
    
    if row['Financial Distress'] < -0.5:
        distress_dict[company][time] = 1
    else:
        distress_dict[company][time] = 0
        
    for feat_idx, column in enumerate(row[3:]):
        feat = feature_names[feat_idx]
        datadict[company][time][feat] = column
        
# print('Dict population complete. Sample below:')
# print("\nData for company 1, time 1:")
# print(datadict[1][1])

# print("\nDistress history for company 1:")
# print(distress_dict[1])

print('We can encode categorical feature 80 as a one-hot vector with this many dimensions:')
print(len(list(set(f80))))

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(max(f80)))
f80_oh = label_binarizer.transform(f80)

# print(f80_oh[0:5])


# ### Génération des données

# In[17]:


# Make new features as np array. We'll even add x80 back!

def rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods):

    for company in range(1, total_n+1):
            
            all_periods_exist = True
            for j in range(0, lookback_periods):
                if not time-j in distress_dict[company]:
                    all_periods_exist = False
            if not all_periods_exist:
                continue
            
            distress_at_eop = distress_dict[company][time]
            new_row = [company]

            for feature in feature_names:
                if feature == 'x80':
                    continue
                feat_sum = 0.0
                variance_arr = []
                for j in range(0, lookback_periods):
                    feat_sum += datadict[company][time-j][feature]
                    variance_arr.append(datadict[company][time-j][feature])
                new_row.append(feat_sum)
                new_row.append(np.var(variance_arr))
                
            for j in range(0,len(f80_oh[0])):
                new_row.append(f80_oh[company-1][j])

            if len(new_row) == ((len(feature_names)-1)*2 + 1 + len(f80_oh[0])) : # we have a complete row
                new_row.append(distress_at_eop)
                new_row_np = np.asarray(new_row)
                train_array.append(new_row_np)
    

def custom_timeseries_cv(datadict, distress_dict, feature_names, total_n, val_time, test_time, 
                         lookback_periods, total_periods=14):

    # Train data
    train_array = []
    for _t in range(1, val_time+1):
        time = (val_time+1) -_t # Start from time period 10 and work backwards
        train_array_np = rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    train_array_np = np.asarray(train_array)
    print(train_array_np.shape)
    # print(train_array_np[0])
    
    # Val data
    if val_time != test_time:
        val_array = []
        for time in range(val_time+1, test_time+1):
            val_array_np = rolling_operation(time, val_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

        val_array_np = np.asarray(val_array)
        print(val_array_np.shape)
        # print(val_array_np[0])
    else:
        val_array_np = None

    # Test data
    test_array = []
    # start from time period 11 and work forwards
    for time in range(test_time+1,total_periods+1):
        test_array_np = rolling_operation(time, test_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    test_array_np = np.asarray(test_array)
    print(test_array_np.shape)
    # print(test_array_np[0])
    
    return train_array_np, val_array_np, test_array_np

# Generate our sets
train_array_np, val_array_np, test_array_np = custom_timeseries_cv(datadict, distress_dict, feature_names, total_n,
                                                     val_time=9, test_time=12, lookback_periods=3, total_periods=14)


# In[18]:


X_train = train_array_np[:,0:train_array_np.shape[1]-1]
y_train = train_array_np[:,-1].astype(int)

X_val = val_array_np[:,0:val_array_np.shape[1]-1]
y_val = val_array_np[:,-1].astype(int)

X_test = test_array_np[:,0:test_array_np.shape[1]-1]
y_test = test_array_np[:,-1].astype(int)

np.set_printoptions(threshold=sys.maxsize)
print(X_train[0,:])
print(y_train)

print(X_val[0,:])
print(y_val)

print(X_test[0,:])
print(y_test)


# In[19]:



from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def model_trial(model_type, hyperparam):
    if model_type in ['logistic-regression']:
        # Logistic Regression. Try 11, l2 penalty, understand one-vs-rest vs multinomial (cross-entropy) 
        model = LogisticRegression(penalty=hyperparam, solver='saga', max_iter=4000)
    elif model_type in ['decision-tree']:
        model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
    elif model_type in ['random-forest']:
        model = RandomForestClassifier(n_estimators=hyperparam)
    else:
        print("Warning: model {} not recognized.".format(model_type))
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print("Mean acc: %f" % model.score(X_val, y_val))
    print("F1: %f" % f1)
    print("Recall: %f" % recall)


# In[20]:


print("-"*20 + "Logistic regression, l1:" + "-"*20)
model_trial('logistic-regression', 'l1')

print("-"*20 + "Logistic regression, l2:" + "-"*20)
model_trial('logistic-regression', 'l2')

print("-"*20 + "Decision tree:" + "-"*20)
model_trial('decision-tree', None)

for i in [2, 4, 10, 50, 100, 1000]:
    print("-"*20 + "Random forest, {} estimators:".format(i) + "-"*20)
    model_trial('random-forest', i)


# In[21]:


#!pip install lightgbm


# ### Les algorithmes de Machine Learning utilisés

# In[22]:


from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier


# In[23]:


knn = KNeighborsClassifier(n_neighbors=15)
clf = knn.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_knb_model=roc_auc_score(y_test, y_pred)*100
acc_knb_model


# In[24]:


lr = LogisticRegression(C = 0.2)
clf1 = lr.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc_log_reg=roc_auc_score(y_test, y_pred1)*100
acc_log_reg


# In[25]:


clf2 = GaussianNB().fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
acc_nb=roc_auc_score(y_test, y_pred2)*100
acc_nb


# In[26]:


clf3 = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt


# In[27]:


clf4 = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
y_pred4 = clf4.predict(X_test)
acc_rmf_model=roc_auc_score(y_test, y_pred4)*100
acc_rmf_model


# In[28]:


clf5 = SVC(gamma='auto').fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)
acc_svm_model=roc_auc_score(y_test, y_pred5)*100
acc_svm_model


# In[29]:


sgd_model=SGDClassifier()
sgd_model.fit(X_train,y_train)
sgd_pred=sgd_model.predict(X_test)
acc_sgd=round(sgd_model.score(X_train,y_train)*100,10)
acc_sgd


# In[30]:


xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
acc_xgb=round(xgb_model.score(X_train,y_train)*100,10)
acc_xgb


# In[31]:


lgbm = LGBMClassifier()
lgbm.fit(X_train,y_train)
lgbm_pred=lgbm.predict(X_test)
acc_lgbm=round(lgbm.score(X_train,y_train)*100,10)
acc_lgbm


# In[32]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Naive Bayes','XGBoost','LightGBM','Decision Tree'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd,acc_nb,acc_xgb,acc_lgbm,acc_dt]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# Le meilleur modèle de prévision de la détresse financière serait ici le gradient stochastique

# In[ ]:




