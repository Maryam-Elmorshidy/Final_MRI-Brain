import streamlit as st
horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 

def result_page():
    
    st.markdown("""
### Interpretation of Results

#### Understanding Prediction Outputs
              """ ,unsafe_allow_html=True)  
    c1,c2 =st.columns(2)
    c1.markdown("""
#### 1. Prediction Outcome:
When analyzing the results generated by our Alzheimer's disease prediction model, it's essential to understand how the predictions are interpreted. Each MRI scan undergoes thorough evaluation, resulting in a prediction that assigns probabilities to four distinct classes:

- **AD (Alzheimer's Disease)**
- **CN (Cognitively Normal)**
- **EMCI (Early Mild Cognitive Impairment)**
- **LMCI (Late Mild Cognitive Impairment)**                
                """ ,unsafe_allow_html=True)
    c2.image("./img/doc1.jpg",width=400)
    st.markdown("""

For instance, a prediction may produce a probability distribution such as [0.2, 0.7, 0.1, 0.0]. Here’s what this distribution signifies:

- **0.2:** This probability indicates the likelihood that the MRI scan exhibits characteristics associated with Alzheimer's Disease (AD).
- **0.7:** This high probability suggests that the MRI scan shows features typical of a Cognitively Normal (CN) state.
- **0.1:** This probability reflects a lower likelihood of Early Mild Cognitive Impairment (EMCI) being present in the MRI scan.
- **0.0:** The absence of probability here indicates that the model assigns a negligible likelihood of Late Mild Cognitive Impairment (LMCI).

Each value represents the model's confidence in classifying the MRI scan into one of these categories based on the patterns and features extracted from the imaging data.

Understanding these prediction outputs is crucial for physicians and healthcare providers to make informed decisions regarding patient care and management strategies.
""" ,unsafe_allow_html=True)
    st.markdown(horizontal_bar, True)
    st.markdown("""
#### 2.Interpreting Probabilities

Understanding the probabilities assigned by our Alzheimer's disease prediction model provides valuable insights into the patient's cognitive health status:

- **Higher Probabilities:**
  When the model assigns higher probabilities to a specific class (e.g., AD, CN, EMCI, LMCI), it indicates a greater likelihood that the MRI scan shows characteristics associated with that stage of Alzheimer's disease. For instance, a high probability for AD suggests significant markers of Alzheimer's disease, prompting further evaluation and potential intervention strategies.

- **Lower Probabilities:**
  Conversely, lower probabilities across all classes indicate that the MRI scan does not prominently exhibit signs of Alzheimer's disease progression. This scenario suggests a healthier cognitive state or less pronounced abnormalities in the imaging data.

These probabilities help clinicians assess the likelihood of Alzheimer's disease stages based on objective data analysis, guiding them in making informed decisions regarding patient care and management.

""" ,unsafe_allow_html=True) 
    st.markdown(horizontal_bar, True)   
    with st.expander("Different Outcomes and Their Meanings") :
        c3,c4 =st.columns(2)
        c3.markdown("""
    #### 1.AD (Alzheimer's Disease)
    - **Interpretation:** A high probability assigned to AD indicates significant markers of Alzheimer's disease observed in the MRI scan, such as characteristic brain changes associated with the disease.
    - **Next Steps for Patients:** It is advisable to recommend further diagnostic tests, such as comprehensive cognitive assessments and additional imaging (e.g., PET scans), to confirm the diagnosis. Discussing treatment options, including medications aimed at managing symptoms and potentially slowing disease progression, is crucial. This discussion should involve the patient and their family members to ensure informed decision-making and support.
    """ ,unsafe_allow_html=True)  
        c4.markdown("""
    #### 2.CN (Cognitively Normal)
    - **Interpretation:** A high probability for CN suggests that the MRI scan displays characteristics consistent with a normal cognitive state, without significant signs of Alzheimer's disease.
    - **Next Steps for Patients:** Reassure the patient about their cognitive health. It’s essential to emphasize the importance of regular cognitive screenings as part of routine healthcare to monitor any changes over time. Encouraging healthy lifestyle choices, including mental and physical activities, can also contribute to maintaining cognitive function.
    """ ,unsafe_allow_html=True) 
        st.image("./img/ADsvCN.png") 
        c5,c6 =st.columns(2)
        
        c5.markdown("""
    #### 3.EMCI (Early Mild Cognitive Impairment)
    - **Interpretation:** A moderate probability for EMCI indicates early signs of cognitive decline that may be indicative of Alzheimer's disease in its initial stages.
    - **Next Steps for Patients:** Recommend a comprehensive neurological evaluation, including detailed memory tests and cognitive assessments. Early detection allows for timely intervention, such as lifestyle modifications (e.g., diet, exercise) and cognitive training programs. Discussing potential treatments, including medications that may slow disease progression, can help manage symptoms effectively.
    """ ,unsafe_allow_html=True)  
        c6.markdown("""
    #### 4.LMCI (Late Mild Cognitive Impairment)
    - **Interpretation:** A moderate to high probability for LMCI indicates more pronounced cognitive impairment compared to EMCI, potentially reflecting progression to an advanced stage of Alzheimer's disease.
    - **Next Steps for Patients:** Initiate sensitive discussions about caregiving options, long-term care planning, and supportive therapies. Coordination with specialists, including neurologists and geriatricians, is crucial to optimize patient care and quality of life. Providing emotional support and resources for both the patient and their caregivers is essential during this challenging time.
                """ ,unsafe_allow_html=True)  
        
        st.image("./img/alzheimers-brain.png")
    st.markdown(horizontal_bar, True)   

    st.markdown("""
### Clinical Decision-Making
#### Confirmatory Tests
""" ,unsafe_allow_html=True)  
    c7,c8 =st.columns(2)
    c7.markdown("""

- **Importance:** It is crucial to emphasize the importance of confirmatory tests and consultations with neurologists or specialists to validate the predictions made by the model.
- **Validation:** While the prediction model provides valuable insights, confirming the diagnosis through additional tests such as PET scans, cerebrospinal fluid analysis, or neuropsychological assessments is essential. These tests help validate the findings and provide a comprehensive understanding of the patient's cognitive health status.
""" ,unsafe_allow_html=True)  
    c8.image("./img/doc.jpg")
    st.markdown(horizontal_bar, True) 
    with st.expander("Patient Counseling") :
        st.markdown("""
    #### Patient Counseling
    - **Personalized Approach:** Each patient and their caregiver(s) may experience varying emotional and practical challenges upon receiving the prediction outcomes.
    - **Emotional Support:** Offer personalized counseling sessions to address the emotional impact of the prediction outcomes. This involves discussing the implications of potential Alzheimer's disease diagnosis, managing anxiety or distress, and providing resources for emotional support.
    - **Practical Guidance:** Provide practical guidance on lifestyle adjustments, caregiving strategies, and planning for future care needs. This includes discussing legal and financial preparations, support services available in the community, and adaptive technologies to enhance daily living.
    """ ,unsafe_allow_html=True)
        st.image("./img/Alzheimer5.png")
    st.markdown(horizontal_bar, True) 
    with st.expander("Longitudinal Monitoring") :
        st.markdown("""
#### Longitudinal Monitoring
- **Need for Follow-Up:** Stress the importance of regular follow-up visits and longitudinal monitoring to assess disease progression or stability over time.
- **Monitoring Plan:** Develop a personalized monitoring plan based on the prediction outcomes and initial diagnostic findings. This plan should include regular cognitive assessments, imaging scans, and evaluations of daily functioning to track changes in cognitive health.
- **Adjustment of Care Plan:** Based on monitoring results, adjust the patient's care plan accordingly. This may involve modifications in medication, therapies, or supportive interventions tailored to the evolving needs of the patient.

""" ,unsafe_allow_html=True)
    st.markdown(horizontal_bar, True)   
    
    st.markdown("""
### Collaborative Approach
""" ,unsafe_allow_html=True)

    with st.expander("Collaborative Approach") :  
        st.markdown("""
#### Multidisciplinary Team
- **Holistic Care:** Alzheimer's disease management requires a collaborative effort involving a multidisciplinary team of healthcare professionals.
- **Team Members:** This team typically includes neurologists, psychologists, social workers, and specialized nurses.
- **Roles and Contributions:** Each member brings unique expertise to address different aspects of Alzheimer's disease care:
  - **Neurologists:** Diagnose and manage neurological aspects of the disease, including treatment options and disease progression.
  - **Psychologists:** Provide behavioral and cognitive assessments, and offer counseling for emotional support.
  - **Social Workers:** Assist with navigating healthcare systems, provide resources for financial and legal planning, and offer support for caregivers.
  - **Specialized Nurses:** Offer medical care, medication management, and guidance on daily living strategies.

#### Patient Education
- **Empowerment Through Knowledge:** Empower patients and their caregivers with comprehensive educational resources and support.
- **Educational Resources:** Provide access to reliable information about Alzheimer's disease, including symptoms, treatment options, and caregiving strategies.
- **Support Groups:** Offer virtual or in-person support groups where patients and caregivers can connect with others facing similar challenges.
- **Informational Sessions:** Host workshops or informational sessions led by healthcare professionals to discuss coping strategies, communication techniques, and advancements in Alzheimer's research.
- **Community Resources:** Collaborate with local organizations and advocacy groups to enhance access to community resources, respite care, and educational workshops.

 """ ,unsafe_allow_html=True)
        st.image("./img/doc3.png") 
    st.markdown(horizontal_bar, True)  
    
    











