import streamlit as st
import random
from PIL import Image
import os 
def home_page():
    horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #749BC2;'><br>" 

    

    vDrive = os.path.splitdrive(os.getcwd())[0]
    vpth = "D:/4AI term 2/project graduation/streamlit/" if vDrive == "C:" else "./"
   
    col1, col2 = st.columns(2)

    # Load a random image
    random.seed()
    image_path = vpth + random.choice(["./img/Alzheimer.jpg","./img/brain1.jpg"])
    image = Image.open(image_path)

    # Display the heading in the first column
    col1.markdown("""<strong style='font-weight: 10000'><span style='font-size: 150%'>Alzheimer's disease (AD)</span></strong> is one of the most common types of neurodegenerative disorders in the aging population. The first signs of AD will typically include forgetfulness and will progress by affecting various functions such as language, motor skills, and memory. However, a slight but noticeable and measurable decline in cognitive abilities, including memory and reasoning abilities, can be associated with mild cognitive impairment (MCI). An individual diagnosed with MCI could be at risk of later developing Alzheimer's, or can be due to age-related memory decline, thus highlighting the importance of early diagnosis of the disease. Still, clinical and neuroimaging studies have demonstrated differences between MCI and normal controls (NC). Patients diagnosed with MCI can be stratified between early MCI (EMCI) and late MCI (LMCI)""",unsafe_allow_html=True)


    # Display the image in the second column with automatic column width
    col2.image(image, use_column_width='auto')

    # thin divider line
    
    st.markdown(horizontal_bar, True)   

    st.markdown("""
### A brief description of each stage:
                                
- ##### **Mild cognitive impairment (MCI):**
    - the affected people are susceptible to forgetting recent occurrences, becoming disoriented in their homes, and having difficulties with communication. This is often the longest stage of AD, lasting up to 4 years."""
        , unsafe_allow_html=False)
    st.markdown("""
- ##### **Early mild cognitive impairment (EMCI):** 
    - the affected person starts experiencing episodes of memory loss with words or the location of household items, nevertheless he can function independently and participate in social activities."""
    , unsafe_allow_html=False)
    c1, c2 , c3 , c4= st.columns(4)
    image_path = vpth + "/img_brain/EMCI/EMCI1.jpg"
    image1 = Image.open(image_path)

    c1.image(image1, use_column_width='auto')

    
    image_path = vpth + "/img_brain/EMCI/EMCI2.jpg"
    image2 = Image.open(image_path)

    c2.image(image2, use_column_width='auto')

    image_path = vpth + "/img_brain/EMCI/EMCI3.jpg"
    image3 = Image.open(image_path)

    c3.image(image3, use_column_width='auto')

    image_path = vpth + "/img_brain/EMCI/EMCI4.jpg"
    image4 = Image.open(image_path)

    c4.image(image4, use_column_width='auto')

    st.markdown(horizontal_bar, True)   

    st.markdown("""
- ##### **Late mild cognitive impairment (LMCI):**
    - at this stage of the disease, patients may need help with daily tasks, facing increasing difficulty communicating and controlling their movements. Their memory and cognitive skills worsen, and changes in behavior and personality may occur."""
        , unsafe_allow_html=False)
    c1, c2 , c3 , c4= st.columns(4)

    image_path = vpth + "/img_brain/LMCI/LMCI1.jpg"
    image1 = Image.open(image_path)

    c1.image(image1, use_column_width='auto')

    
    image_path = vpth + "/img_brain/LMCI/LMCI2.jpg"
    image2 = Image.open(image_path)

    c2.image(image2, use_column_width='auto')

    image_path = vpth + "/img_brain/LMCI/LMCI3.jpg"
    image3 = Image.open(image_path)

    c3.image(image3, use_column_width='auto')

    image_path = vpth + "/img_brain/LMCI/LMCI4.jpg"
    image4 = Image.open(image_path)

    c4.image(image4, use_column_width='auto')

    st.markdown(horizontal_bar, True) 



    st.markdown("""
- ##### **Alzheimer’s disease (AD):**
    - as the disease progresses, the affected person requires increasing levels of attention and aid with daily tasks. This stage is characterized by growing unawareness of time and space, problems recognizing family and close friends, difficulty walking, and behavioral disturbances that may even lead to aggression.""" 
                , unsafe_allow_html=False)
    c1, c2 , c3 , c4= st.columns(4)

    image_path = vpth + "/img_brain/AD/AD1.jpg"
    image1 = Image.open(image_path)

    c1.image(image1, use_column_width='auto')

    
    image_path = vpth + "/img_brain/AD/AD2.jpg"
    image2 = Image.open(image_path)

    c2.image(image2, use_column_width='auto')

    image_path = vpth + "/img_brain/AD/AD3.jpg"
    image3 = Image.open(image_path)

    c3.image(image3, use_column_width='auto')

    image_path = vpth + "/img_brain/AD/AD4.jpg"
    image4 = Image.open(image_path)

    c4.image(image4, use_column_width='auto')

    st.markdown(horizontal_bar, True) 


  

    



#