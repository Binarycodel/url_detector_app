from urllib.parse import urlparse
import re
from tld import get_tld
import joblib
import numpy as np


# FEATURE TO EXTRACT == # url_len , abnormal_url, https, digits, leters, shortining_service, havin_ip, 

class URLExtractFeatures:
   

    
    
    def __init__(self, data):
        self.data = data
        self.url_lenght = len(data)
        print('url lenght : ' , self.url_lenght)

   

    def normal_url(self, url):
        hostname = urlparse(url).hostname
        hostname = str(hostname)
        match = re.search(hostname, url)
        if match:
            # print match.group()
            return 1
        else:
            # print 'No matching pattern found'
            # print(f'Abnormal  URL: {url}   ===================   HOSTNAME: {hostname}')
            return 0
    



    # checking for shortening services 
    def shortining_Service(self, url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          'tr\.im|link\.zip\.net',
                          url)
        if match:
            return 1
        else:
            return 0


    def having_ip_address(self, url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
            '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
            '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
        if match:
            return 1
        else:
            return 0

    
    
    def get_features(self):
        
         # removing www.... 
        remove_www = self.data.replace('www.', '')
        remove_www

        # ectracting symbols features
        feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']

        symbol_feature = []
        for a in feature:
            symbol_feature.append(remove_www.count(a))
        
        print('symbol feature ', symbol_feature)
            
        #checking url if normal or not         
        norm_url =  self.normal_url(self.data)
        print('normal url : ' , bool(norm_url) , norm_url)
        
         # extracting http or https features
        http_secure = 1 if str(urlparse(self.data).scheme)=='https' else 0
        print('http or https : ' , bool(http_secure))

        # extracting digit count 
        digit_count = len([d for d in self.data if d.isnumeric()])
        print('digit_count = ' , digit_count)

        # extracting letter count 
        alpha_count = len([d for d in self.data if d.isalpha()])
        print('alpha_count = ' , alpha_count)
        
        short_service = self.shortining_Service(self.data)
        print('short_service =', bool(short_service))
        
        having_ip = self.having_ip_address(self.data)
        print("having_ip = ", bool(having_ip))
            
        # list variable to hold url extracted features
        features_holder = []
        features_holder.append(self.url_lenght)
        [features_holder.append(f) for f in symbol_feature]
        features_holder.append(norm_url)
        features_holder.append(http_secure)
        features_holder.append(digit_count)
        features_holder.append(alpha_count)
        features_holder.append(short_service)
        features_holder.append(having_ip)
        
        return features_holder
    

class Predictor: 
    def __init__(self):
        self.models = []
    
    def load_model(self ):
        model_paths = ['model/url_adaboost_model.jb',  'model/url_classifier_randomforest_model.jb', 'model/url_SGD_model.jb']
        m1 = joblib.load(model_paths[0])
        m2 = joblib.load(model_paths[1])
        m3 = joblib.load(model_paths[2])
        self.models = [m1, m2, m3]
        print('models' , self.models)
        return self.models
       
        
    def essemble_prediction(self, features):

        prediction = [model.predict([features])[0] for model in self.models]
        print('prediction : ',  prediction)
        final_pre = np.bincount(prediction).argmax()
        print(prediction)
        print('final prediction : = ' , final_pre)
        return final_pre , prediction
