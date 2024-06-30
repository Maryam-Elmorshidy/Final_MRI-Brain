import streamlit as st
horizontal_bar_big = "<hr style='margin-top: 0; margin-bottom: 0; height: 3px; border: 3px solid #749BC2;'><br>" 

def model_page():
    
    st.markdown("""
### Welcome, Doctors!

#### Understanding Our Prediction Model

Welcome to our webpage, where we offer a comprehensive overview of our cutting-edge prediction model for diagnosing Alzheimer's disease. As medical professionals, we understand the importance of clarity and accuracy when it comes to utilizing predictive tools for patient care.

#### What You'll Find Here:

**Model Overview:**
   - We provide detailed insights into how our prediction model works, including its underlying architecture, data sources, and the methodologies employed in its development.
   - **Data Collection:**
     - **MRI Scans:** The foundation of our prediction model lies in the extensive use of MRI (Magnetic Resonance Imaging) scans. MRI is a non-invasive imaging technique that provides detailed images of the brain's structure. This detailed visualization is crucial for identifying subtle changes in the brain associated with Alzheimer's disease.
     - **Focus on Brain Structure:** The model specifically analyzes MRI scans that capture detailed views of the brain's structure. Key areas of interest include the hippocampus, cortical thickness, white matter integrity, and the presence of amyloid plaques or neurofibrillary tangles.
     - **High-Quality Data Sources:** The dataset is sourced from well-known medical imaging repositories, ensuring high-quality and reliable data.

**Clinical Relevance:**
   - Explore how our prediction model can aid in early diagnosis and treatment planning for Alzheimer's disease. We highlight its potential impact on clinical practice and patient care.

#### Why It Matters:

Our prediction model isn't just about numbers and algorithms—it's about improving patient outcomes and empowering healthcare professionals like you to make informed decisions. By understanding how our model works and its clinical implications, you can better integrate it into your practice and provide optimal care for your patients.

#### Join Us in the Fight Against Alzheimer's:

Alzheimer's disease is a complex condition with significant implications for patients and their families. With your support and collaboration, we can continue to refine and optimize our prediction model, ultimately making a meaningful difference in the lives of those affected by this devastating disease.

""",unsafe_allow_html=True)
    st.markdown(horizontal_bar_big, True)




        














