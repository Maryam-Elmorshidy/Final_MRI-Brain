# calling libraries
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
import warnings
warnings.filterwarnings("ignore")
from navigation.function import *


# Define global variables
horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'>"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the trained model to check
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('navigation/simple_mri_classifier.pth', map_location=device))
model.eval()
model_ = model.to(device)    
def patient_page():
    horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 
    c1,c2 = st.columns(2)

    c1.markdown("""
### Introduction:

Alzheimer’s disease (AD) is the most common neurological disease due to a disorder known as cognitive impairment, which progresses to deterioration in cognitive abilities, behavioral changes, and memory loss; AD affects the adaptability that needs to be promoted. Despite the scientific advancement in the medical field, there is no active cure for AD, but the effective method for AD is to slow its progression. Therefore, early detection of Alzheimer’s symptoms in its first stage is vital to prevent its progression to advanced stages. Dementia is one of the most common forms of AD due to the lack of effective treatment for the disease. AD progresses slowly before clinical biomarkers appear.

""",unsafe_allow_html=True)
    
    c2.image("./img/Alzheimer6.jpg")
    c2.markdown("""
                
In our quest to aid early diagnosis and intervention, we have developed a prediction model that utilizes advanced technology to analyze MRI (Magnetic Resonance Imaging) scans of the brain. MRI scans provide detailed images of the brain's structure and function, allowing us to identify subtle changes associated with Alzheimer's disease.                

Our prediction model harnesses the power of these MRI scans to assess the risk of developing Alzheimer's disease in individuals. By analyzing specific features and patterns in the brain captured by MRI, the model can provide valuable insights into a person's likelihood of experiencing cognitive decline associated with Alzheimer's disease.

In the following sections, we'll delve into how our prediction model works, the significance of early detection, and how patients can understand the results of their MRI scan predictions. We aim to empower individuals and their families with knowledge and resources to navigate their journey with Alzheimer's disease more effectively.

--- 


""",unsafe_allow_html=True)
    
    c1.image("./img/Alzheimer4.jpg")


    st.markdown(horizontal_bar, True) 

    st.markdown("# **prediction of MRI brain**", unsafe_allow_html=True)
    with st.expander("let's go "):

        st.markdown("""
    ### Classification of MRI brain image to know type of Alzheimer's disease
    ---""" , unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an MRI brain image...")
        st.markdown("\n", unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            prediction = predict_image(image, model_, device)
            
            if prediction == 0:
                st.write("This is an MRI image.")
                model_path = 'navigation/alzahimer_resnet50_model.sav'
                num_classes = 4

                try:
                    model = load_torchscript_model(model_path, device)
                except RuntimeError:
                    model = load_state_dict_model(model_path, num_classes, device)

                image_tensor = preprocess_image(image)            
                prediction, probabilities = predict(model, image_tensor, device)

                # Display uploaded image
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Uploaded MRI Image', use_column_width=True)


                # Display probabilities as a horizontal bar chart with color gradients
                with col2:
                    st.write("Class Probabilities:")
                    classes = ['AD', 'CN', 'EMCI', 'LMCI']
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(classes, probabilities, color=plt.cm.get_cmap('viridis')(probabilities))
                    ax.set_xlabel('Probability')
                    ax.set_title('Class Probabilities')
                    ax.invert_yaxis()  # Invert y-axis for better visualization
                    for bar, prob in zip(bars, probabilities):
                        ax.text(prob, bar.get_y() + bar.get_height() / 2, f'{prob:.2f}', va='center', ha='right', color='white')
                    st.pyplot(fig)


                # Identify the class with the highest probability
                max_prob_index = np.argmax(probabilities)
                st.write(f"The image is classified as: {classes[max_prob_index]} with a probability of {probabilities[max_prob_index]:.2f}")
        #         type_disease = classes[max_prob_index]
        #         if type_disease == 'AD':
        #             st.markdown("""
        # #### AD (Alzheimer's Disease)
        # - **Interpretation:** A high probability assigned to AD indicates significant markers of Alzheimer's disease observed in the MRI scan, such as characteristic brain changes associated with the disease.
        # - **Next Steps for Patients:** It is advisable to recommend further diagnostic tests, such as comprehensive cognitive assessments and additional imaging (e.g., PET scans), to confirm the diagnosis. Discussing treatment options, including medications aimed at managing symptoms and potentially slowing disease progression, is crucial. This discussion should involve the patient and their family members to ensure informed decision-making and support.
        # """ ,unsafe_allow_html=True) 
        #         elif type_disease == 'EMCI':
        #             st.markdown("""
        # #### EMCI (Early Mild Cognitive Impairment)
        # - **Interpretation:** A moderate probability for EMCI indicates early signs of cognitive decline that may be indicative of Alzheimer's disease in its initial stages.
        # - **Next Steps for Patients:** Recommend a comprehensive neurological evaluation, including detailed memory tests and cognitive assessments. Early detection allows for timely intervention, such as lifestyle modifications (e.g., diet, exercise) and cognitive training programs. Discussing potential treatments, including medications that may slow disease progression, can help manage symptoms effectively.
        # """ ,unsafe_allow_html=True)
        #         elif type_disease == 'LMCI':
        #             st.markdown("""
        # #### LMCI (Late Mild Cognitive Impairment)
        # - **Interpretation:** A moderate to high probability for LMCI indicates more pronounced cognitive impairment compared to EMCI, potentially reflecting progression to an advanced stage of Alzheimer's disease.
        # - **Next Steps for Patients:** Initiate sensitive discussions about caregiving options, long-term care planning, and supportive therapies. Coordination with specialists, including neurologists and geriatricians, is crucial to optimize patient care and quality of life. Providing emotional support and resources for both the patient and their caregivers is essential during this challenging time.
        #             """ ,unsafe_allow_html=True)
        #         elif type_disease == 'CN':
        #             st.markdown("""
        # #### CN (Cognitively Normal)
        # - **Interpretation:** A high probability for CN suggests that the MRI scan displays characteristics consistent with a normal cognitive state, without significant signs of Alzheimer's disease.
        # - **Next Steps for Patients:** Reassure the patient about their cognitive health. It’s essential to emphasize the importance of regular cognitive screenings as part of routine healthcare to monitor any changes over time. Encouraging healthy lifestyle choices, including mental and physical activities, can also contribute to maintaining cognitive function.
        # """ ,unsafe_allow_html=True) 
                

            else:
                st.write("This is not an MRI image.")
                c1, c2 = st.columns(2)
                c1.image(image, caption='Uploaded Image', use_column_width=True)
       
    

        st.markdown(horizontal_bar, True)
    st.markdown(horizontal_bar, True) 
    st.markdown("# **For more information**", unsafe_allow_html=True)
    with st.expander("Explanation of MRI Scans"):
        st.markdown("""
### Explanation of MRI Scans

##### *What is an MRI Scan??*

MRI stands for Magnetic Resonance Imaging. It's a safe and non-invasive technique that doctors use to get detailed images of the inside of your body, including your brain.
Think of an MRI scan like taking a photograph of your brain, but instead of using a camera, it uses powerful magnets and radio waves.

##### *How Does It Work?*

The MRI machine creates a strong magnetic field around your head. This field, along with radio waves, interacts with the water molecules in your brain to produce signals.
These signals are captured by the machine and processed by a computer to create detailed images of your brain's structure and function.

##### *Why is it Important?*

These images help doctors see changes in your brain that might be related to Alzheimer's disease. By looking at these images, doctors can identify patterns that suggest whether Alzheimer's might be present.

                
##### *Can an MRI Diagnose Alzheimer’s?*

The simplest answer to the question is yes. The more complicated answer considers that there is still a lot of research to do on this disease, so it may be a while before we establish a definitive test to diagnose Alzheimer’s disease.
However, for the time being, using an MRI to detect Alzheimer’s is one of the best options available.

##### Easy to Understand Analogy:

Imagine your brain is like a city. An MRI scan is like taking a high-resolution aerial photo of that city, allowing you to see the roads, buildings, and parks clearly. Similarly, an MRI gives doctors a clear picture of the different parts of your brain and how they're working.


""" , unsafe_allow_html=True)
        st.image("./img/MRI.jpg")
        st.markdown(horizontal_bar, True) 

    
    with st.expander("Role of MRI Scans in Predicting Alzheimer's Disease"):
        st.markdown("""
### Role of MRI Scans in Predicting Alzheimer's Disease

**How MRI Scans Help:**
- MRI scans give doctors a detailed look at your brain's structure and health. These images show different parts of your brain and how they're working.
- When it comes to Alzheimer's disease, certain changes in the brain can be early signs of the condition. MRI scans can help detect these changes.
""" , unsafe_allow_html=True)
    
        c3,c4 = st.columns(2)
        c3.markdown("""
**What the Prediction Model Looks For:**
- **Brain Volume:** One of the things the prediction model looks at is the size of different parts of your brain. In Alzheimer's disease, some areas of the brain may shrink. The model can detect these changes in brain volume.
- **Abnormal Protein Deposits:** The model also looks for signs of abnormal protein deposits, like amyloid plaques and tau tangles, which are often found in the brains of people with Alzheimer's.
- **Brain Connectivity:** The model examines how different parts of your brain communicate with each other. Changes in brain connectivity can be another sign of Alzheimer's disease.
""" , unsafe_allow_html=True)
    
        c4.image("./img/Alzheimer5.jpg")
    
        st.markdown("""
**How It Works:**
- The prediction model uses advanced algorithms to analyze the MRI scans. Think of it like a very smart detective that looks for clues in the images.
- By identifying specific patterns and features in the MRI scans, the model can estimate the likelihood that a person might have or develop Alzheimer's disease.

**Why This Matters:**
- Understanding these changes early on allows doctors to make informed decisions about your health. Early detection can lead to better planning and treatment options, potentially slowing the progression of the disease and improving your quality of life.


""" , unsafe_allow_html=True)
    
    st.markdown(horizontal_bar, True) 
    
    