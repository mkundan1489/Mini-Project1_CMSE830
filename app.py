import pandas as pd
import altair as alt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
#from imblearn.over_sampling import SMOTENC,RandomOverSampler,KMeansSMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
sns.set()


data  = pd.read_csv('hypothyroid.csv')
data = data.drop(['TBG'], axis=1)
data = data.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured'], axis=1)
for column in data.columns:
    count = data[column][data[column]=='?'].count()
    if count!=0:
        data[column] = data[column].replace('?',np.nan)    
        
# We can map the categorical values like below:
data['sex'] = data['sex'].map({'F' : 0, 'M' : 1})

# except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
# so instead of mapping indvidually, let's do a smarter work
for column in data.columns:
    if  len(data[column].unique())==2:
        data[column] = data[column].map({'f' : 0, 't' : 1})
        
# this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories. 
data = pd.get_dummies(data, columns=['referral_source'])

lblEn = LabelEncoder()

data['Class'] =lblEn.fit_transform(data['Class'])

imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
new_array=imputer.fit_transform(data) # impute the missing values
    # convert the nd-array returned in the step above to a Dataframe
new_data=pd.DataFrame(data=np.round(new_array), columns=data.columns)

final_data = new_data[new_data['age'] <= 100]

final_data['Class'] = final_data['Class'].map({1 : 'negative', 0 : 'compensated_hypothyroid',
                                              2: 'primary_hypothyroid', 3:'secondary_hypothyroid' })
final_data = final_data.drop(['referral_source_STMW', 'referral_source_SVHC','referral_source_SVHD','referral_source_SVI','referral_source_other'],axis =1)

data = data.drop(['referral_source_STMW', 'referral_source_SVHC','referral_source_SVHD','referral_source_SVI','referral_source_other'],axis =1)
new_data = new_data.drop(['referral_source_STMW', 'referral_source_SVHC','referral_source_SVHD','referral_source_SVI','referral_source_other'],axis =1)

intr = st.button("Intoduction to Thyroid Dataset")
if intr:
    st.write("Thyroid is a dataset for detection of thyroid diseases, in which patients diagnosed with hypothyroid or neagtive.")
    image = Image.open('ThyroidAnatomy.webp')
    st.image(image, caption='Thyroid Gland')
    st.write("The dataset is obtained from the Kaggle The Pupose of the this webb app is to understand the dataset and visualize its independent and dependent features.")
    st.write("Check the boxes on the left side to know more about them.")
st.sidebar.header('THYROID DATASET')
st.sidebar.subheader('Check the box if you want to understand about dataset or Visualize:')
ud = st.sidebar.checkbox('Understand Dataset')
if ud:
    st.header("Below is the Thyroid dataset:")
    st.dataframe(final_data)
    st.write('Shape of the dataset is: ',final_data.shape)
    st.subheader("Select below buttons to understand about them:")
    cl =  st.button('FEATURES')
    ms = st.button("MISSING")
    im = st.button("IMPUTE")
    ot = st.button("OUTLIERS")
    fn = st.button("FINAL DATA")
    if cl:
        st.write("age: Age of the patients")
        st.write("sex: Sex of the the patients")
        st.write("on_thyroxine: Wheteher the patient is on thyroxine")
        st.write("on_antithyroid_medication: Whether the patient is on antithyroid medication")
        st.write("sick: Whetehr the patient is sick")
        st.write("pregnant: Whetehr the patient is pregnant")
        st.write("thyroid_surgery: Whether the patient has undergone thyroid surgery")
        st.write("goitre: Whetehr the patient has goitre")
        st.write("tumor: Whether the patient has tumor")
        st.write("hypopituitary: whether patient has hypopituitary")
        st.write("psych: whether patient has psych")
        st.write("TSH: TSH (Thyroid Stimulating Hormone) level in the blodd(0.005 to 530)")
        st.write("T3: T3 level in the blood(0 to 11)")
        st.write("TT4: TT4(Thyroxine) level in the blood(2 to 430)")
        st.write("T4U: T4 Uptake in the blood level(0.25 to 2.12)")
        st.write("FIT: Thyroxine(T4)/Thyroid Binding Capacity (2 to 395)")
        st.write("Class: Negative or types of hypothyroid")
    if ms:
        missing_data = data.isna().sum()
        st.write("These are the number of misssing data for each columns:", missing_data)
        st.write("Here the heatmap for the same")
        fig4 = plt.figure(figsize=(10,6))
        sns.heatmap(data.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
        st.pyplot(fig4)
        st.write("We can visualize from above that missing data is MAR. Below graph gives the histogram of missing data.")
        plt.figure(figsize=(10,6))
        fig5 =  sns.displot(
            data=data.isna().melt(value_name="missing"),
            y="variable",
            hue="missing",
            multiple="fill",
            aspect=1.25
)
        st.pyplot(fig5)
    if im:
        st.write("KNN imputer is used to fill the missing data.")
        st.write(new_data.isna().sum())
    if ot:
        st.write("Now lets look at outliers:")
        fig6=plt.figure(figsize =(10, 6))
        plt.boxplot(new_data)
        plt.show()
        fig6
        st.write("From above boxplot we can see that we have outlier.")
    if fn:
        st.write("Full dataset has been studied and missing values are replaced. And also, outliers have been taken care of. Below is the final data.")
        st.write("This dataset has been used for visualizations.")
        st.dataframe(final_data)
        
        


vz = st.sidebar.checkbox('Visualization')
if vz: 
    st.text('We will see the distribution of hormone levels in the blood for each of our classes in the dataset.')
    option = st.sidebar.select_slider(
        'Please slide to select different hormone below:',
        ('TSH', 'T3', 'TT4', 'T4U', 'FTI'))

    st.sidebar.write('You selected:', option)

#fig=sns.stripplot(data=final_data, x="Class", y=option)
#st.pyplot(fig)
#plt.show()

    fig=alt.Chart(final_data).mark_point().encode(
        x='Class',
        y=option,
        color='Class',
        shape='Class'
    ).properties(width = 600, height = 600).interactive()
    st.altair_chart(fig)
    st.write("After Visualizing all the hormones distribution:")
    st.write("We can hypothesize that FTI, T3, and TT4 will be good feature additions to our models. TSH looks like it might be good as well but we need to handle the outliers for class hypo and analyze the attribute distribution further before making any decisions. This is all in-line with the knowledge discovered about Hormone level tests during my initial research.")


    option2 = st.sidebar.multiselect(
        'Please select multiple options:',
        ('age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'))
    for i in range(len(option2)):
        fig2=alt.Chart(final_data).mark_circle().encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color='Class:N'
        ).properties(
        width=300,
        height=300
        ).repeat(
            row=[option2[i]],
            column=[option2[i] for i in range(len(option2))]
        ).interactive()
        st.altair_chart(fig2)
        i=i+1
    if len(option2)>0:
        st.subheader("Observations:")
        st.write("We can see that for some Hormone test vs others there are nice clusters that form. This is encouragin because it means that they do a good job at separating out each of our target classes.")
        st.write("FTI vs T3")
        st.write("FTI vs T4U")
        st.write("FTI vs age")
        st.write("T4U vs TT4")
        st.write("TT4 vs age")
        st.write("TT4 vs T3")


