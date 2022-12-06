import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')

st.title('Streamlit Basics')


def main():
	activites = ['EDA' , 'Visualisation' , 'Model' , 'About Us']
	option = st.sidebar.selectbox('Select option:',activites)

	if option == 'EDA':
		#EDA
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader('Upload dataset : ',type = ['csv','xlsv','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Display shape'):
				st.write(df.shape)

			if st.checkbox("Display columns"):
				st.write(df.columns)

			if st.checkbox("Select multiple columns"):
				selected_columns = st.multiselect('Select preferred columns',df.columns)
				df1 = df[selected_columns]
				st.dataframe(df1)

			if st.checkbox("Display summary"):
				st.write(df.describe().T)

			if st.checkbox('Display Null values'):
				st.write(df.isnull().sum())

			if st.checkbox('Display the data Types'):
				st.write(df.dtypes)

			if st.checkbox("Display Correlation of data various columns"):
				st.write(df.corr())

	elif option == 'Visualisation':
		st.subheader('Visualisation')
		data = st.file_uploader('Upload dataset : ',type = ['csv','xlsv','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select multiple columns to plot'):
				selected_columns = st.multiselect('Select your preferred columns',df.columns)
				df1= df[selected_columns]
				st.dataframe(df1)

			if st.checkbox('Heatmap'):
				fig = plt.figure()
				st.write(sns.heatmap(df.corr(),vmax = 1,square = True , annot = True,cmap ='viridis'))
				st.pyplot(fig)

			if st.checkbox('Display Pairplot'):
				#fig1 = plt.figure()
				st.write(sns.pairplot(df,diag_kind = 'kde'))
				st.pyplot()

			if st.checkbox('Display Pie chart'):
				all_columns = df.columns.to_list()
				pie_columns = st.selectbox('Select columns to display',all_columns)
				piechart = df[pie_columns].value_counts().plot.pie(autopct = '%1.1f%%')
				st.write(piechart)
				st.pyplot()

	elif option == 'Model':
		st.subheader('Model Building')
		data = st.file_uploader('Upload dataset : ',type = ['csv','xlsv','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox('Select multiple columns'):
				new_data = st.multiselect('Select your preferred columns',df.columns)
				df1= df[new_data]
				st.dataframe(df1)

				#Dividing my data into x and y variables

				x = df1.iloc[: , 0:-1]
				y = df1.iloc[: , -1]

			seed = st.sidebar.slider('Seed' ,1,200)

			classifier_name = st.sidebar.selectbox('Select your prefered classifier: ',('KNN' , 'SVM' , 'LR' ,  'Decision tree'))

			def add_parameter(name_of_clf):
				params = dict()
				if name_of_clf == 'SVM':
					C = st.sidebar.slider('C',0.01,15.0)
					params['C'] = C
				else:
					K = st.sidebar.slider('K',1,15)
					params['K'] = K
				return params

			params = add_parameter(classifier_name)

			#defining a function for our classifier

			def get_classifier(name_of_clf,params):
				clf = None
				if name_of_clf == 'SVM':
					clf = SVC(C = params['C'])
				elif name_of_clf == 'KNN':
					clf = KNeighborsClassifier(n_neighbors = params['K'])
				elif name_of_clf == 'LR':
					clf = LogisticRegression()
				# elif name_of_clf == 'Naive_Bayes':
				# 	clg = GaussianNB()
				elif name_of_clf == 'Decision tree':
					clf = DecisionTreeClassifier()
				else:
					st.warning('Select your choce of algorithm')

				return clf

			clf = get_classifier(classifier_name,params)

			x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state=seed)

			clf.fit(x_train,y_train)

			y_pred = clf.predict(x_test)
			st.write('Predcitions',y_pred)

			accuracy = accuracy_score(y_test,y_pred)

			st.write('Name of classifier',classifier_name)
			st.write('Accuracy',accuracy)

	elif option=="About Us":
		st.subheader("About us")
		st.write("Voila!!! We have successfully created our ML App")
		st.write("This is our ML App to make things eaier for our users to understand their data without a stress")


		st.balloons()

if __name__ == '__main__':
	main()