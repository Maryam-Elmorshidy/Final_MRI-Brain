import streamlit as st
import hydralit_components as hc
import streamlit as st

# import streamlit_analytics
from navigation.model_for_DR import model_page
from navigation.using_modelDR import use_page
from navigation.results import result_page

try:
    from streamlit import rerun as rerun
except ImportError:
    # conditional import for streamlit version <1.27
    from streamlit import experimental_rerun as rerun

horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 
horizontal_bar_big = "<hr style='margin-top: 0; margin-bottom: 0; height: 3px; border: 3px solid #749BC2;'><br>" 

def doctor_page():
    st.markdown("""
 ### Introduction to the Doctor's Portal
    
    **Welcome to the Doctor's Portal:**
    
    This portal serves as a comprehensive tool for healthcare professionals involved in the diagnosis and management of Alzheimer's disease using MRI scans. Our advanced prediction model leverages cutting-edge technology to provide accurate assessments and insights into cognitive health.
    
    Whether you are a neurologist, radiologist, or general practitioner, this platform is designed to streamline the integration of MRI-based predictions into your clinical workflow. Here’s what you can explore on this page:
    """ ,unsafe_allow_html=True)
    

    MODEL = "Our Model"
    USE = "Use"
    RESULT = "Result"
    RESOURCES = "Resources"

    tabs =[ MODEL , USE , RESULT , RESOURCES]

    option_data =[
        {'icon': "⚙🔎", 'label': MODEL},
        {'icon': "🛠", 'label': USE},
        {'icon': "🥇", 'label': RESULT},
        
    ]
    over_theme = {'txc_inactive': 'black', 'menu_background': '#D6E5FA', 'txc_active': 'white', 'option_active': '#749BC2'}
    #font_fmt = {'font-class': 'h3', 'font-size': '50%'}

    chosen_tab =hc.option_bar(option_definition= option_data ,
                              title="",
                              key="SeconderyOption",
                              override_theme= over_theme,
                              horizontal_orientation= True)
    if chosen_tab == MODEL:
        model_page()

    if chosen_tab == USE:
        use_page()    

    if chosen_tab == RESULT: 
        result_page()  

   




