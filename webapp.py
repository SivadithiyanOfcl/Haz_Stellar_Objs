import streamlit as st
import pickle
#import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
import pandas as pd

df= pd.read_csv("scaler_data.csv")

sample = [
[1.1982708007,2.6794149658,13569.2492241812,54839744.08284605,16.73],
[0.2658,0.5943468684,73588.7266634981,61438126.52395093,20.0],
[0.7220295577,1.6145071727,114258.6921290512,49798724.94045679,17.83],
[0.096506147,0.2157943048,24764.3031380016,25434972.72075825,22.2],
[0.2550086879,0.5702167609,42737.7337647264,46275567.00130072,20.09],
[0.0363542322,0.0812905344,34297.5877783029,40585691.22792288,24.32],
[0.1716148941,0.3837425691,27529.4723069673,29069121.41864897,20.95],
[0.0053278866,0.0119135167,57544.4700827352,55115019.25807114,28.49],
[0.3503926411,0.7835017643,56625.2101223615,69035980.03881611,19.4],
[0.2526707542,0.5649889822,58430.6971996129,38337496.948336646,20.11]]

knn_model = pickle.load(open('knn.pkl','rb'))
svc_model = pickle.load(open('svc.pkl','rb'))
naive_model = pickle.load(open('nb.pkl','rb'))
gbc_model = pickle.load(open('gbc.pkl','rb'))
tree_model = pickle.load(open('tree.pkl','rb'))
forest_model = pickle.load(open('forest.pkl','rb'))
nb_model = pickle.load(open('nb.pkl','rb'))

sc = StandardScaler()
df = sc.fit_transform(df)

def classify(num):
    if num==0:
        return 'Not Hazardous'
    else:
        return 'Hazardous'

def sample_input():
    
    n = random.randint(1,5)
    st.session_state.field1 = str(sample[n][0])
    st.session_state.field2 = str(sample[n][1]) 
    st.session_state.field3 = str(sample[n][2]) 
    st.session_state.field4 = str(sample[n][3]) 
    st.session_state.field5 = str(sample[n][4])

def main():
    
    st.title("Hazardous Stellar object Classifier")
    
    html_temp = """
    <div style="background-color:yellow ;padding:10px">
    <h2 style="color:black;text-align:center;">Hazardous Object Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Gradient Boosting Classifier','Decision Tree','Support Vector Classifier','Random Forest Classifier',
               'Naive Bayes Classifier','K Nearest Neighbors Classifier']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
 
    min_dia=st.text_input("Enter Minimum Diameter in km👇",placeholder="eg. 1.1982708007",key="field1")
    max_dia=st.text_input("Enter Maximum Diameter in km👇",placeholder="eg. 2.6794149658",key="field2")
    velocity=st.text_input("Enter Relative Velocity 👇",placeholder="eg. 13569.2492241812",key="field3")
    dist=st.text_input("Enter Distance from Earth in km👇",placeholder="eg. 54839744.08284605",key="field4")
    abs_mag=st.text_input("Enter Absolute Magnitude 👇",placeholder="eg. 16.73",key="field5")

    c1, c2 = st.columns(2)
    with c1:
        add = st.button(label="Generate Sample Values",on_click = sample_input)
    with c2:
        classifier = st.button('Classify')

    if classifier:
        lst = [float(min_dia), float(max_dia), float(velocity), float(dist), float(abs_mag)]
        inputs=sc.transform([lst])
        if option=='Gradient Boosting Classifier':
            st.success(classify(gbc_model.predict(inputs)))
        elif option=='Decision Tree':
            st.success(classify(tree_model.predict(inputs)))
        elif option=='Naive Bayes Classifier':
            st.success(classify(nb_model.predict(inputs)))
        elif option=='Support Vector Classifier':
            st.success(classify(svc_model.predict(inputs)))
        elif option=='Random Forest Classifier':
            st.success(classify(forest_model.predict(inputs)))
        elif option=='K Nearest Neighbors Classifier':
            st.success(classify(knn_model.predict(inputs)))
        else:
            st.success(classify(forest_model.predict(inputs)))

if __name__=='__main__':
    main()
