import hydralit_components as hc
import streamlit as st
import os

from navigation.doctor import doctor_page
from navigation.home import home_page
from navigation.patient import patient_page
from navigation.more import more_page

from utils.components import footer_style, footer

try:
    from streamlit import rerun as rerun
except ImportError:
    # conditional import for streamlit version <1.27
    from streamlit import experimental_rerun as rerun

# Determine the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(
    page_title="Alzheimer's Tech",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)


max_width_str = f"max-width: {75}%;"

st.markdown(f"""
        <style>
        .appview-container .main .block-container{{{max_width_str}}}
        </style>
        """,
            unsafe_allow_html=True,
            )

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    
                }
        </style>
        """, unsafe_allow_html=True)


###NavBar###

HOME = 'Home'
PATIENT = 'Patient'
DOCTOR = 'Doctor'
Palestine_Flag = 'Free Palestine'


tabs = [
    HOME,
    PATIENT,
    DOCTOR,
    Palestine_Flag,
 
]

option_data = [
    {'icon': "🏠", 'label': HOME},
    {'icon': "🤕", 'label': PATIENT},
    {'icon': "👨‍⚕️", 'label': DOCTOR},
    {'icon': "🇵🇸", 'label': Palestine_Flag},
    
]

over_theme = {'txc_inactive': 'black', 'menu_background': '#D6E5FA', 'txc_active': 'white', 'option_active': '#749BC2'}
font_fmt = {'font-class': 'h3', 'font-size': '50%'}

chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    override_theme=over_theme,
    horizontal_orientation=True)


if chosen_tab == HOME:
    home_page()

elif chosen_tab == PATIENT:
    patient_page()

elif chosen_tab == DOCTOR:
    doctor_page()

elif chosen_tab == Palestine_Flag:
    more_page()    



# Footer

st.markdown(footer_style, unsafe_allow_html=True)

for i in range(4):
    st.markdown('#')
st.markdown(footer, unsafe_allow_html=True)


###sidebar###

sidebar_image_path = os.path.join(ROOT_DIR, "img", "Palestine Flag10.gif")
st.sidebar.image(sidebar_image_path, use_column_width=10,) 




st.sidebar.title('Alzheimer\'s Disease Prediction')
st.sidebar.markdown('Welcome to the Alzheimer\'s Disease Prediction app!')

st.sidebar.subheader('About')
st.sidebar.info('This app uses deep learning models to predict the likelihood of Alzheimer\'s disease based on MRI brain scans.')

st.sidebar.subheader('Instructions')
st.sidebar.write('1. Upload an MRI brain image using the file uploader on the doctor page (or patient page).')
st.sidebar.write('2. The app will analyze the image and provide predictions along with interpretations.')
st.sidebar.write('3. You can learn more about MRI Brain in the patient page.')


st.sidebar.subheader('Contact Us')
st.sidebar.write('For inquiries or support, please contact:')
st.sidebar.markdown('- Email: contact@alzheimerprediction.com')
st.sidebar.markdown('- Phone: +1 (123) 456-7890')

st.sidebar.subheader('Disclaimer ')
st.sidebar.write('This app is for educational purposes only. Consult a healthcare professional for medical advice.')


# Adjust the path to the sidebar image
sidebar_image_path = os.path.join(ROOT_DIR, "img", "Alzheimer4.png") 
st.sidebar.image(sidebar_image_path, use_column_width=True) 

