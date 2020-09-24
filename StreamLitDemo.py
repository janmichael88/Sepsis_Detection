import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#https://github.com/ShivamBhirud/Capital-Bike-Share-Data-Streamlit-Web-Application/blob/master/demoStreamlit.py
#https://docs.streamlit.io/en/stable/tutorial/create_a_data_explorer_app.html
#https://docs.streamlit.io/en/stable/api.html#display-charts
#https://discuss.streamlit.io/t/drop-down-menu/3180



test = 'demo_test_csv.csv'

#CACHE FUNCTIONS

@st.cache
def load_data_and_process():
	data = pd.read_csv(test, index_col=0)
	return(data)

data_load_state = st.text('Loading data...')
data = load_data_and_process()
data_load_state.text("Done! (using st.cache)")



#UN CAHCED

def pre_process(frame,length=100,method='linear'):
	#subset frame
    frame = frame.iloc[:,list(range(2,37))+[-1]]
    #get size
    rows,cols = frame.shape[0],frame.shape[1]
    #broadcast last_row
    last_row = frame.iloc[-1,:]
    for i in range(length-rows):
        frame = frame.append(last_row,ignore_index=True)
        #first interpolat linearly
    frame = frame.interpolate(method=method)
    #then backfill,forwardfill, then 0
    frame = frame.fillna(method='bfill').fillna(method='ffill').fillna(0)
    #assert nans
    assert np.sum(pd.isnull(frame.values)) == 0, 'There are still NaNs!'
    assert frame.shape[0] == length,'Not the right size!'
    return(frame)

def plot_sepsis_covariates(col_name,data,col1,col2):
    #create the above plot a colname
    fig,ax = plt.subplots(figsize=(20, 5))
    # make a plot
    ax.plot(data['Delta_t_hours'], data[col_name], color=col1, marker="o")
    # set x-axis label
    ax.set_xlabel("Hours",fontsize=14)
    # set y-axis label
    ax.set_ylabel(col_name,color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(data['Delta_t_hours'], data['SepsisLabel'],color=col2,marker="o")
    ax2.set_ylabel("Sepsis Onset",color="blue",fontsize=14)
    plt.title(col_name)
    st.pyplot(plt)

def pull_xy(frame):
	x = frame.iloc[:,:-1].values
	y = frame.iloc[:,-1].values
	return(x,y)    

def instantiate_model():
	model = load_model('LSTM_hiddenstates_v2.h5')
	return(model)


def plotly_two(col_name,data):
	# Create figure with secondary y-axis
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# Add traces
	fig.add_trace(
	    go.Scatter(x=data['Delta_t_hours'], y=data[col_name], name=col_name),
	    secondary_y=False,
	)

	fig.add_trace(
	    go.Scatter(x=data['Delta_t_hours'], y=data['SepsisLabel'], name="Sepsis Onset"),
	    secondary_y=True,
	)

	# Add figure title
	fig.update_layout(
	    title_text="Double Y Axis Example"
	)

	# Set x-axis title
	fig.update_xaxes(title_text="xaxis title")

	# Set y-axes titles
	fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
	fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)
	st.plotly_chart(fig)

def plotly_plot_predictions(output):
	x = list(range(0,100))
	fig = go.Figure(data=go.Scatter(x=x, y=output))

	st.plotly_chart(fig)



#main call
def main():
	st.title('ICU patient Septic Shock Detection and Data Explorer')

	#checkbox to show raw data
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")	
		st.write(data)

	#second checkbox	
	if st.checkbox('Process Data? This will remove Nans,forwardfill and interpolate accordingly'):
		st.subheader('Preprocessing....')
		st.write(pre_process(data))

	#checkbox to examine time series
	if st.checkbox('Examine Time Series and Sepsis Onset Together?'):
		option = st.selectbox('What covariates to you wish to compare?',
			tuple(data.columns))
		plotly_two(option,data)

	#checkbox to create predictino vector
	if st.checkbox('Predict Probability of Septic Shock for this patient'):
		x,y = pull_xy(pre_process(data))
		#predict
		model = instantiate_model()
		output = model.predict(x.reshape((1,100,35)))
		plotly_plot_predictions(output[0].ravel().tolist())
		#plotly_plot_predictions(output)


if __name__ == "__main__":
	main()