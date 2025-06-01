import streamlit as st
import pandas as pd
import mlflow


left_column, right_column, top, bottom = st.columns(4)
with left_column:
    having_IP_Address = st.selectbox('having_IP_Address',['Yes' ,'No'])
    URL_Length = st.selectbox('URL_Length', [ 'Yes','Not Found', 'No'])
    Shortining_Service = st.selectbox('Shortining_Service',[ 'Yes', 'No'])
    having_At_Symbol = st.selectbox('having_At_Symbol',[ 'Yes', 'No'])
    double_slash_redirecting = st.selectbox('double_slash_redirecting', ['No','Yes'])
    Prefix_Suffix = st.selectbox('Prefix_Suffix', ['No', 'Yes'])
    having_Sub_Domain = st.selectbox('having_Sub_Domain', ['No', 'Not Found', 'Yes'])
    SSLfinal_State = st.selectbox('SSLfinal_State' , ['No','Yes','Not Found'])
    

with right_column:
    Domain_registeration_length = st.selectbox('Domain_reg_length', ['No' , 'Yes'])
    Favicon = st.selectbox('Favicon' , [ 'Yes', 'No'])
    port = st.selectbox('port', [ 'Yes' ,'No'])
    HTTPS_token = st.selectbox('HTTPS_token' , ['No', 'Yes'])
    Request_URL = st.selectbox('Request_URL' , [ 'Yes' ,'No'])
    URL_of_Anchor = st.selectbox('URL_of_Anchor' , ['No', 'Not Found', 'Yes'])
    Links_in_tags = st.selectbox('Links_in_tags' , [ 'Yes' ,'No' , 'Not Found'])
    SFH = st.selectbox('SFH' , ['No' ,'Yes' ,'Not Found'])

with top:
    Submitting_to_email = st.selectbox('Submitting_to_email', ['No' , 'Yes'])
    Abnormal_URL = st.selectbox('Abnormal_URL' , [ 'Yes', 'No'])
    Redirect = st.selectbox('Redirect', [ 'Yes' ,'Not Found'])
    on_mouseover = st.selectbox('on_mouseover' , ['No', 'Yes'])
    RightClick = st.selectbox('RightClick' , [ 'Yes' ,'No'])
    popUpWidnow = st.selectbox('popUpWidnow' , ['No', 'Yes'])
    Iframe = st.selectbox('Iframe' , [ 'Yes' ,'No' ])
    age_of_domain = st.selectbox('age_of_domain' , ['No' ,'Yes' ])

with bottom:
    DNSRecord = st.selectbox('DNSRecord', ['No' , 'Yes'])
    web_traffic = st.selectbox('web_traffic' ,['No' , 'Not Found' ,'Yes'])
    Page_Rank = st.selectbox('Page_Rank', [ 'Yes' ,'No'])
    Google_Index = st.selectbox('Google_Index' , ['No', 'Yes'])
    Links_pointing_to_page = st.selectbox('Links_pointing_to_page' , [ 'Yes','Not Found' ,'No'])
    Statistical_report = st.selectbox('Statistical_report' , ['No', 'Yes'])

dic = {
    'having_IP_Address' : having_IP_Address,
    'URL_Length' : URL_Length,
    'Shortining_Service' : Shortining_Service,
    'having_At_Symbol' : having_At_Symbol,
    'double_slash_redirecting' : double_slash_redirecting,
    'Prefix_Suffix' : Prefix_Suffix,
    'having_Sub_Domain' : having_Sub_Domain,
    'SSLfinal_State' : SSLfinal_State,
    'Domain_registeration_length' : Domain_registeration_length,
    'Favicon' : Favicon,
    'port' : port,
    'HTTPS_token' : HTTPS_token,
    'Request_URL' : Request_URL,
    'URL_of_Anchor' : URL_of_Anchor,
    'Links_in_tags' : Links_in_tags,
    'SFH' : SFH,
    'Submitting_to_email' : Submitting_to_email,
    'Abnormal_URL' : Abnormal_URL,
    'Redirect' : Redirect,
    'on_mouseover' : on_mouseover,
    'RightClick' : RightClick,
    'popUpWidnow' : popUpWidnow,
    'Iframe' : Iframe,
    'age_of_domain' : age_of_domain,
    'DNSRecord' : DNSRecord,
    'web_traffic' : web_traffic,
    'Page_Rank' : Page_Rank,
    'Google_Index' : Google_Index,
    'Links_pointing_to_page' :Links_pointing_to_page,
    'Statistical_report' :Statistical_report}
dataframe = pd.DataFrame(dic, index =[0])
for column in dataframe.columns:
    dataframe[column] = dataframe[column].replace({
    'Yes': 1,
    'No': -1,
    'Not found': 0
})

def prediction(dataframe):
    # Set the tracking URI to the local mlruns directory
    mlflow.set_tracking_uri(uri ="http://localhost:5000")

    # Now load the model
    model = mlflow.sklearn.load_model("models:/best_model/latest")
    result = model.predict(dataframe)
    return result

if st.button ("The final Prdiction"):
    result = prediction(dataframe)
    # result
    if result == 1:
        st.write("The website is fake")
    else: st.write("The website is not fake")

    