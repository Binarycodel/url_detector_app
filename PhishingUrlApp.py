import streamlit as st 
import url_detector as dt


st.header('Assemble Based Malicious URL Detector')


# pass feature into predictor 
predictor = dt.Predictor()





with st.spinner('model looding.....'):
    machine_model = predictor.load_model() # loading machine learning models


url_data = st.text_input('Paste URL... ')
st.write(url_data)

# extracting url features..... 
url_extract = dt.URLExtractFeatures(url_data)
features = url_extract.get_features()




rem = {"Category": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}

if  st.button("analyze"):
    # perform prediction ...
    if features != []: 

        with st.spinner('Predicting.......'):
            result, predic = predictor.essemble_prediction(features)
   
        
        result_score = {
            "AdaBosst" : predic[0] , 
            "RandomForest" : predic[1], 
            "SGD" : predic[2]
        }

        st.write(rem)
        st.subheader('prediction')
        st.write(result_score)
        st.subheader('Final Prediction')
        st.write(f'Final Prediction =   {result}')
        
        if url_data != '':
            if result == 0 :
                st.success('URL is Benign (Not Hamful)')
            elif result == 1 : 
                st.warning("defacement URL (Hamful)")
            elif result == 2 : 
                st.error("fishing URL (Dangerious)")
            else :
                st.error("Malware URL (Dangerious)")
        else:
            st.error("kindly supply valid URL")
    else : 
        st.error('No Feature Detected')