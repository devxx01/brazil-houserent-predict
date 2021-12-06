import pickle
import streamlit as st
 
# loading the trained model
import joblib
regressor = joblib.load('model.pkl')
 
@st.cache()

# predictors = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'floor', 'animal', 'furniture']
# defining the function which will make the prediction using the data which the user inputs 
def prediction(city, area, rooms, bathroom, parking_spaces, floor, animal, furniture):
  # Preprocessing
  # city
  if city == 'Belo Horizonte':
    city =  0
  elif city == 'Campinas':
    city = 1
  elif city == 'Porto Alegre':
    city = 2
  elif city == 'Rio de Janeiro':
    city = 3
  elif city == 'São Paulo':
    city = 4
  # animal
  if animal == 'Accept':
    animal = 1
  elif animal == 'Not Accept':
    animal = 0
  # furniture
  if furniture == 'Furnished':
    furniture = 1
  elif furniture == 'Not Furnished':
    furniture = 0
  
  # Making prediction

  X = [[(city), int(area), int(rooms), int(bathroom), int(parking_spaces), int(floor), (animal), (furniture)]]
  prediction = regressor.predict(X)
  
  return prediction

  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:LightBlue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Brazil House Rent Prediction</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    # following lines create boxes in which user can enter data required to make prediction 
    city = st.selectbox('City',("Belo Horizonte","Campinas","Porto Alegre","Rio de Janeiro","São Paulo"))
    area = st.slider("Area", 1, 300) 
    rooms = st.number_input("Rooms", 1, 10)
    bathroom = st.number_input('Bathroom',1, 10)
    parking_spaces = st.number_input("Parking Spaces", 1, 10) 
    floor = st.number_input('Floor', 1, 20)
    animal = st.radio("Animal",['Accept','Not Accept'])
    furniture = st.radio("Furniture",['Furnished','Not Furnished'])
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction((city), int(area), int(rooms), int(bathroom), int(parking_spaces), int(floor), (animal), (furniture))
        st.success('House Rent Price is {}'.format(result))
     
if __name__=='__main__': 
    main()