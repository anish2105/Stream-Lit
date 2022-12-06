import streamlit as st

st.title("Our first StreamLit App")

from PIL import Image

st.subheader("this is a sub")
image  = Image.open('art1.JPG')
st.image(image,use_column_width = True)
st.write('Wrtitng here')
st.markdown("Markdown that as a comment")
st.success("Success")
st.info("this is a info")
st.error("Oop error")
st.warning("be cautous")
st.help(range)

import numpy as np
import pandas as pd

df = np.random.rand(10,20)
st.dataframe(df)

st.text("---" * 100)

df1 = pd.DataFrame(np.random.rand(10,20) , columns = ('col %d'  %i  for i in range(20)))
st.dataframe(df1.style.highlight_max(axis=1))

st.text("---" * 100)

chart_data = pd.DataFrame(np.random.randn(20,3) , columns = ['a' , 'b' , 'c'])
st.line_chart(chart_data)

st.area_chart(chart_data)

chart_data1 = pd.DataFrame(np.random.randn(50,3) , columns = ['a' , 'b' , 'c'])
st.bar_chart(chart_data1)

import matplotlib.pyplot as plt

arr = np.random.normal(1,1,size = 100)
plt.hist(arr,bins = 20)
st.pyplot()

st.text("---" * 100)

import plotly
import plotly.figure_factory as ff

x1 = np.random.randn(200)-2
x2 = np.random.randn(200)
x3 = np.random.randn(200)-2

hist_data = [x1,x2,x3]
group_labels = ['Group1' , 'Group2' , 'Group3']
fig = ff.create_distplot(hist_data , group_labels,bin_size = [.2,.25,.5])
st.plotly_chart(fig,use_container_width = True)

st.text("---" * 100)

df = pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.56-122.4] , columns = ['lat' , 'lon'])
st.map(df)

#creating buttons

if st.button("Say Hello"):
	st.write("Hello is here")
else:
	st.write("Y??")

st.text("---" * 100)

genre = st.radio('what is your genre' , ('Comedy','Drama','Documentary'))

if genre == 'Comedy':
	st.write("Oh nice")
elif genre == 'Drama':
	st.write("Drama HMM")
else:
	st.write("Boring")

st.text("---" * 100)

#select button

option = st.selectbox("How was your night?" , ('Fantasic' , 'Osm' , 'Okayish'))

st.write("Your ans was:",option)

st.text("---" * 100)

option1 = st.multiselect("How was your night select a option?" , ('Fantasic' , 'Osm' , 'so'))

st.write("Your ans was:",option1)

st.text("---" * 100)

age  = st.slider('How old are u? ', 0,150,10)
st.write("Your age is : ", age)


values = st.slider('Select a range of values' , 0,200,(15,80))
st.write("You selected a range between :",values)

num = st.number_input('Input Number')
st.write("The number you chose is" , num)

st.text("---" * 100)

#file operator

upload_file = st.file_uploader("choose a csv file" , type ='csv')

if upload_file is not None:
	data = pd.read_csv(upload_file)
	st.write(data)
	st.success('Successfully')
else:
	st.markdown("Please upload a csv file")

#color picker

# color = st.beta_color_picker("Pick your preferred color:" , '#00f900')
# st.write("This is your color: ",color)

#Slide bar

st.text("---" * 100)

add_sidebar = st.sidebar.selectbox("What is your fav course?" , ('ds','others','otherone' , 'thisone'))

import time

my_bar = st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)



with st.spinner('wait for it....'):
	time.sleep(1)
st.success('Successful')

st.balloons()