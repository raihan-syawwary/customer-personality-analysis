#!/usr/bin/env python
# coding: utf-8

# # Customer Personality Analysis 
# 
# #### The analysis is conducted to assist the business with its customers data, in order to facilitate in modifying products according to the specific needs, behaviours, and concerns of different types of customers. The objective of this analysis is to cluster the customers based on their personalities, so the business will be able to determine their opinions about the product, and what kind of actions they take rather than their sayings about the product.  

# In[1]:


pip install mlxtend


# ## Import Packages

# In[2]:


import numpy as np
import pandas as pd
import datetime 
import os
from datetime import date
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')


# ## Data Overview

# In[3]:


import csv

with open('marketing_campaign.csv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')  
    for row in reader:
        print(row) 


# In[4]:


input_file = 'marketing_campaign.csv'
output_file = 'marketing_campaign_updated.csv'

with open(input_file, 'r') as file_in, open(output_file, 'w', newline='') as file_out:
    reader = csv.reader(file_in, delimiter='\t')
    writer = csv.writer(file_out, delimiter=',')

    for row in reader:
        writer.writerow(row)


# In[5]:


data=pd.read_csv('marketing_campaign_updated.csv', header=0, sep=',')


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data['Marital_Status'].unique()


# In[9]:


from dataprep.eda import plot, plot_correlation, create_report, plot_missing
plot(data)


# ## Data Preparation

# In[10]:


#Handling Some Features and Generating New Features 

#1. Age
data['Age']=2023-data['Year_Birth']

#2. Spending
data['Spending'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

#3. Seniority 
last_date = date(2023, 7, 3)
data['Seniority'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True, format='%d-%m-%Y')
data['Seniority']=pd.to_numeric(data['Seniority'].dt.date.apply(lambda x: (last_date - x)).dt.days, downcast='integer')/30

#4. Marital Status (handling unique values)
data['Marital_Status']=data['Marital_Status'].replace({'Divorced':"Alone", 'Single':"Alone", 'Married':"In Couple", 'Together':"In Couple", 'Absurd':"Alone", 'Widow': "Alone", 'YOLO':"Alone"})

#5. Education (handling unique values)
data['Education']=data['Education'].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Postgraduate', 'Master':'Postgraduate', 'PhD': 'Postgraduate'})

#6. Children 
data['Children']=data['Kidhome']+data['Teenhome']

#7. Has_Child 
data['Has_child'] = np.where(data.Children>0, 'Yes', 'No')

data=data.rename(columns={'NumWebPurchases':'Web', 'NumCatalogPurchases':'Catalog', 'NumStorePurchases':'Store'})
data=data.rename(columns={'MntWines': 'Wines', 'MntFruits':'Fruits', 'MntMeatProducts':'Meat', 'MntFishProducts':'Fish', 'MntSweetProducts': 'Sweets', 'MntGoldProds':'Gold'})

data=data[['ID', 'Age', 'Education', 'Marital_Status', 'Income', 'Spending', 'Seniority', 'Has_child', 'Children', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']]

data.head()


# In[11]:


data.info()


# ### Data Quality Check

# In[12]:


#missing values percentage checking
data.isna().mean()*100


# In[13]:


data['Income'].unique()


# In[14]:


#outliers checking 
data['Income'].plot(kind='box')
plt.title('Income', size=16)
plt.show()


# In[15]:


#dropping rows where its 'Income' column has missing value
data=data.dropna(subset=['Income'])


# In[16]:


#handling outliers with clipping
Q1 = data['Income'].quantile(0.25)
Q3 = data['Income'].quantile(0.75)
IQR = Q3 - Q1 
Lwhisker = Q1 - 1.5 * IQR 
Uwhisker = Q3 + 1.5 * IQR 
data['Income_clipped'] = data['Income'].clip(Lwhisker, Uwhisker) 


# In[17]:


data['Income_clipped'].plot(kind='box')
plt.title('Boxplot Income (clipped)', size=16)
plt.show()


# In[18]:


data['Income_clipped'].unique()


# In[19]:


data.info()


# ### Clustering

# In[20]:


scaler=StandardScaler()
sets_temp=data[['Income_clipped', 'Seniority', 'Spending']]
X_std=scaler.fit_transform(sets_temp)
X = normalize(X_std,norm='l2')

gmm=GaussianMixture(n_components=4, covariance_type='spherical', max_iter=2000, random_state=5).fit(X)
labels = gmm.predict(X)
sets_temp['Cluster'] = labels 
sets_temp = sets_temp.replace({0:'Stars', 1:'Need Attention', 2:'Highly Potential', 3:'Leaky Bucket'})
data=data.merge(sets_temp.Cluster, left_index=True, right_index=True)
pd.options.display.float_format ="{:.0f}".format 


# In[21]:


summary=data[['Income_clipped', 'Spending', 'Seniority', 'Cluster']]
summary.set_index("Cluster", inplace = True)
summary=summary.groupby('Cluster').describe().transpose()
summary.head()


# In[22]:


PLOT = go.Figure()
for C in list(data.Cluster.unique()):
    PLOT.add_trace(go.Scatter3d(x = data[data.Cluster == C]['Income'],
                                y = data[data.Cluster == C]['Seniority'],
                                z = data[data.Cluster == C]['Spending'],
                                mode = 'markers', marker_size = 6, marker_line_width = 1,
                                name = str(C)))
PLOT.update_traces(hovertemplate='Income: %{X} <br>Seniority: %{y} <br>Spending: %{z}')
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'Income', titlefont_color = 'black'),
                                yaxis=dict(title = 'Seniority', titlefont_color = 'black'),
                                zaxis=dict(title ='Spending', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color = 'black', size = 12))


# ### Data Preparation for CPA 

# In[23]:


#Age Segment
cut_labels_age = ['Young', 'Adult', 'Mature', 'Senior']
cut_bins = [0, 30, 45, 65, 120]
data['Age Group'] = pd.cut(data['Age'], bins=cut_bins, labels=cut_labels_age)

#Income Segment
cut_labels_income = ['Low Income', 'Low to Medium Income', 'Medium to High Income', 'High Income']
data['Income Group'] = pd.qcut(data['Income_clipped'], q=4, labels=cut_labels_income)

#Seniority Segment
cut_labels_seniority = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
data['Seniority Group'] = pd.qcut(data['Seniority'], q=4, labels=cut_labels_seniority)

data=data.drop(columns=['Age', 'Income', 'Seniority', 'Income_clipped'])
data.head()


# In[24]:


#Spending-on-each-product segment 
cut_labels = ['Low Buyer', 'Frequent Buyer', 'Biggest Buyer']
data['Wines_Segment'] = pd.qcut(data['Wines'][data['Wines']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data['Fruits_Segment'] = pd.qcut(data['Fruits'][data['Fruits']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data['Meat_Segment'] = pd.qcut(data['Meat'][data['Meat']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data['Fish_Segment'] = pd.qcut(data['Fish'][data['Fish']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data['Sweets_Segment'] = pd.qcut(data['Sweets'][data['Sweets']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data['Gold_Segment'] = pd.qcut(data['Gold'][data['Gold']>0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
data.replace(np.nan, "Non buyer", inplace=True)
data.drop(columns=['Spending', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'], inplace=True)
data=data.astype(object)
data.head()


# ### Apriori Algorithm

# In[25]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 999)
pd.options.display.float_format = "{:.3f}".format
association=data.copy()
df=pd.get_dummies(association)
min_support = 0.08
max_len = 10
frequent_items = apriori(df, use_colnames=True, min_support=min_support, max_len=max_len + 1)
rules = association_rules(frequent_items, metric='lift', min_threshold=1)


# In[42]:


#Wines Biggest Customers
product='Wines'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_wines = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_wines.head()


# In[27]:


results_personal_care_wines.info()


# In[28]:


antecedents_counts = results_personal_care_wines['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# In[29]:


#Fruits Biggest Customers
product='Fruits'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_fruits = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_fruits.head()


# In[30]:


results_personal_care_fruits.info()


# In[31]:


antecedents_counts = results_personal_care_fruits['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# In[32]:


#Meat Biggest Customers
product='Meat'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_meat = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_meat.head()


# In[33]:


antecedents_counts = results_personal_care_meat['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# In[34]:


#Fish Biggest Customers
product='Fish'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_fish = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_fish.head()


# In[35]:


antecedents_counts = results_personal_care_fish['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# In[36]:


#Sweets Biggest Customers
product='Sweets'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_sweets = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_sweets.head()


# In[37]:


antecedents_counts = results_personal_care_sweets['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# In[38]:


#Gold Biggest Buyers
product='Gold'
segment = 'Biggest Buyer'
target = '{\'%s_Segment_%s\'}' %(product,segment)
results_personal_care_gold = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personal_care_gold.head()


# In[39]:


antecedents_counts = results_personal_care_gold['antecedents'].value_counts()
print(antecedents_counts.head())
most_common_value = antecedents_counts.idxmax()
print("Most Common Value:", most_common_value)


# ### Summary Report - Conclusion

# #### Based on analysis run on July 3rd, 2023, the results are shown below:
# 
# #### 1. The biggest buyers of wines are mostly old customers and those who in "Need attention" cluster 
# 
# #### 2. The biggest buyers of fruits are mostly biggest buyers of sweets and biggest buyers of fish 
# 
# #### 3. The biggest buyers of meat are mostly high-incomed customers, customers who don't have any children, and biggest buyers of fish. 
# 
# #### 4. The biggest buyers of sweets are mostly from "Need attention" customers cluster, biggest buyers of fish, and biggest buyers of fruits 
# 
# #### 5. The biggest buyers of gold are mostly from "Need attention" customers, and biggest buyers of fish. 

# ###

# ### Personal Care Table Convert to Excel

# In[40]:


excel_file_path = 'C:\\Users\\ASUS\\OneDrive\\Documents\\Raihan\\Personal Care\\Personal Care.xlsx'
excel_writer = pd.ExcelWriter(excel_file_path)
results_personal_care_wines.to_excel(excel_writer, sheet_name='Wines Personal Care', index=False)
results_personal_care_fruits.to_excel(excel_writer, sheet_name='Fruits Personal Care', index=False)
results_personal_care_meat.to_excel(excel_writer, sheet_name='Meat Personal Care', index=False)
results_personal_care_fish.to_excel(excel_writer, sheet_name='Fish Personal Care', index=False)
results_personal_care_sweets.to_excel(excel_writer, sheet_name='Sweets Personal Care', index=False)
results_personal_care_gold.to_excel(excel_writer, sheet_name='Gold Personal Care', index=False)
excel_writer.save()

print("Data exported to Excel successfully!")

