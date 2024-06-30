#libraries
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from navigation.function import * 

# Define global variables
horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'>"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the trained model to check
model = models.resnet18(pretrained=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('navigation/simple_mri_classifier.pth', map_location=device))
model.eval()
model_ = model.to(device)    


# Streamlit app interface
def use_page():
    st.markdown("""
    ### Classification of MRI brain image to determine type of Alzheimer's disease
    """, unsafe_allow_html=True)
    st.markdown(horizontal_bar, True)

    uploaded_file = st.file_uploader("Choose an MRI brain image...")
    st.markdown("\n", unsafe_allow_html=True)

    if uploaded_file is not None:
        image_upload = Image.open(uploaded_file)
        prediction = predict_image(image_upload, model_, device)
        
        if prediction == 0:
            st.write("This is an MRI image.")
            model_path = 'navigation/alzahimer_resnet50_model.sav'
            num_classes = 4

            try:
                model = load_torchscript_model(model_path, device)
            except RuntimeError:
                model = load_state_dict_model(model_path, num_classes, device)

            image_tensor = preprocess_image(image_upload)            
            prediction, probabilities = predict(model, image_tensor, device)

            # Display uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_upload, caption='Uploaded MRI Image', use_column_width=True)


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
            type_ = classes[max_prob_index]
            st.write(f"The image is classified as: {classes[max_prob_index]} with a probability of {probabilities[max_prob_index]:.2f}")
            
    #         if type_ == 'AD':
    #             st.markdown("""
    # #### AD (Alzheimer's Disease)
    # - **Interpretation:** A high probability assigned to AD indicates significant markers of Alzheimer's disease observed in the MRI scan, such as characteristic brain changes associated with the disease.
    # - **Next Steps for Patients:** It is advisable to recommend further diagnostic tests, such as comprehensive cognitive assessments and additional imaging (e.g., PET scans), to confirm the diagnosis. Discussing treatment options, including medications aimed at managing symptoms and potentially slowing disease progression, is crucial. This discussion should involve the patient and their family members to ensure informed decision-making and support.
    # """ ,unsafe_allow_html=True) 
    #         elif type_ == 'EMCI':
    #             st.markdown("""
    # #### EMCI (Early Mild Cognitive Impairment)
    # - **Interpretation:** A moderate probability for EMCI indicates early signs of cognitive decline that may be indicative of Alzheimer's disease in its initial stages.
    # - **Next Steps for Patients:** Recommend a comprehensive neurological evaluation, including detailed memory tests and cognitive assessments. Early detection allows for timely intervention, such as lifestyle modifications (e.g., diet, exercise) and cognitive training programs. Discussing potential treatments, including medications that may slow disease progression, can help manage symptoms effectively.
    # """ ,unsafe_allow_html=True)
    #         elif type_ == 'LMCI':
    #             st.markdown("""
    # #### LMCI (Late Mild Cognitive Impairment)
    # - **Interpretation:** A moderate to high probability for LMCI indicates more pronounced cognitive impairment compared to EMCI, potentially reflecting progression to an advanced stage of Alzheimer's disease.
    # - **Next Steps for Patients:** Initiate sensitive discussions about caregiving options, long-term care planning, and supportive therapies. Coordination with specialists, including neurologists and geriatricians, is crucial to optimize patient care and quality of life. Providing emotional support and resources for both the patient and their caregivers is essential during this challenging time.
    #             """ ,unsafe_allow_html=True)
    #         elif type_ == 'CN':
    #             st.markdown("""
    # #### CN (Cognitively Normal)
    # - **Interpretation:** A high probability for CN suggests that the MRI scan displays characteristics consistent with a normal cognitive state, without significant signs of Alzheimer's disease.
    # - **Next Steps for Patients:** Reassure the patient about their cognitive health. It’s essential to emphasize the importance of regular cognitive screenings as part of routine healthcare to monitor any changes over time. Encouraging healthy lifestyle choices, including mental and physical activities, can also contribute to maintaining cognitive function.
    # """ ,unsafe_allow_html=True)         


        else:
            st.write("This is not an MRI image.")
            c1, c2 = st.columns(2)
            c1.image(image_upload, caption='Uploaded Image', use_column_width=True)

    
    st.markdown(horizontal_bar, True)


if __name__ == '__main__':
    use_page()



