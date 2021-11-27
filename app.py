from os import name
from flask import Flask,render_template, request
from flask_mysqldb import MySQL
#from flaskext.mysql import MySQL

import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher

from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import pickle5 as pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras import metrics
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers import Flatten

from datetime import datetime

from nltk import word_tokenize, pos_tag, chunk
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from pprint import pprint
import pandas as pd
import numpy as np

import tensorflow as tf
from keras import optimizers
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.constraints import maxnorm
from keras.layers import Dropout
import os
from pathlib import Path
import json

from werkzeug.utils import secure_filename
 
app = Flask(__name__,template_folder='template',static_url_path='/static')
 
app.config['MYSQL_HOST'] = 'database-1.ckrlvqpkgb22.us-east-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_PASSWORD'] = 'Orangee11'
app.config['MYSQL_DB'] = 'AWS_RDS_SCHEMA-1'

 
mysql = MySQL(app)
 
# @app.route('/forms', methods =["GET", "POST"])
# def form():
#     return render_template('form.html')

@app.route('/index', methods =["GET", "POST"])
def index():
    return render_template('index-3.html')

@app.route('/chatbot', methods =["GET", "POST"])
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot-home', methods =["GET", "POST"])
def chatbot_home():
    base_path = "static/data.json"
    fd = open(base_path , 'r')

    with open(base_path) as json_data:
        data_dict = json.load(json_data)
    
    json_acceptable_string = data_dict.replace("""'""", r"""\"""")
    print(json_acceptable_string)
    d = json.loads(json_acceptable_string)

    # print()

    candidate_name = d['local-user']['name']
    location = d['local-user']['location']
    email_address = d['local-user']['email']
    phone_country_code = "+" + d['local-user']['p_country_code']
    phone = d['local-user']['phone_number']
    date_of_birth = (d['local-user']['dob'])[:2] + "-" + (d['local-user']['dob'])[2:4] + "-" + (d['local-user']['dob'])[4:]
    candidate_qualification = d['local-user']['qualification']
    candidate_availability =  d['local-user']['availability']
    experience = int(d['local-user']['experience'])
    about_candidate = d['local-user']['about_me']

    with open(r"/home/ubuntu/encoder_model.pkl", "rb") as input_file:
        encoder = pickle.load(input_file)

    new_model = tf.keras.models.load_model('/home/ubuntu/static/cnn_model')

    encoded_docs = [one_hot(about_candidate, 1000)]
    padded_text = pad_sequences(encoded_docs, maxlen=1000, padding='post')
    prediction = new_model.predict(padded_text)
    result = encoder.inverse_transform(prediction)
    model_prediction = result[0]

    cursor = mysql.connection.cursor()
    cursor.execute(''' INSERT INTO candidate (can_about,can_availability,can_dob,can_email,can_experience,can_location,can_name,can_phone,can_phone_ctry_code,can_qualification,recommended_jobs) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(about_candidate, candidate_availability, date_of_birth, email_address, experience, location, candidate_name, phone, phone_country_code, candidate_qualification, model_prediction))
    mysql.connection.commit()
    cursor.close()
    val = "Details Entered Successfully" + candidate_name

    os.remove("/home/ubuntu/static/data.json")

    return render_template('dashboard.html')

@app.route('/logical-assessment', methods =["GET", "POST"])
def logical_assessment():
    return render_template('logical-assessment.html')

@app.route('/logical-assessment-send', methods =["GET", "POST"])
def logical_assessment_send():
    if request.method == 'POST':
        q1 = request.form.get('q1')
        q2 = request.form.get('q2')
        q3 = request.form.get('q3')
        q4 = request.form.get('q4')
        q5 = request.form.get('q5')
        q6 = request.form.get('q6')
        q7 = request.form.get('q7')
        q8 = request.form.get('q8')
        q9 = request.form.get('q9')
        q10 = request.form.get('q10')
        q11 = request.form.get('q11')
        q12 = request.form.get('q12')
        q13 = request.form.get('q13')
        q14 = request.form.get('q14')
        q15 = request.form.get('q15')
        q16 = request.form.get('q16')
        q17 = request.form.get('q17')
        q18 = request.form.get('q18')
        q19 = request.form.get('q19')

        answers = ['1-A','2-E','3-B','4-D','5-C','6-A','7-E','8-A','9-A','10-B','11-B','12-E','13-E','14-D','15-A','16-E','17-B','18-E','19-D']
        questions = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19]

        score = 0
        for i in range(len(questions)):
            for j in range(len(answers)):
                if i == j:
                    if questions[i] == answers[i]:
                        score = score + 1
        
        score = round((score/len(questions)*100),2)

        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE candidate SET logical_score = " + str(score) + " WHERE can_id = 11113")
        mysql.connection.commit()
        cursor.close()
    return render_template('personality-assessment.html')

@app.route('/personality-assessment', methods =["GET", "POST"])
def personality_assessment():
    return render_template('personality-assessment.html')

@app.route('/personality-assessment-send', methods =["GET", "POST"])
def personality_assessment_send():
    if request.method == 'POST':
        q1 = int(request.form.get('q1'))
        q2 = int(request.form.get('q2'))
        q3 = int(request.form.get('q3'))
        q4 = int(request.form.get('q4'))
        q5 = int(request.form.get('q5'))
        q6 = int(request.form.get('q6'))
        q7 = int(request.form.get('q7'))
        q8 = int(request.form.get('q8'))
        q9 = int(request.form.get('q9'))
        q10 = int(request.form.get('q10'))
        q11 = int(request.form.get('q11'))
        q12 = int(request.form.get('q12'))
        q13 = int(request.form.get('q13'))
        q14 = int(request.form.get('q14'))
        q15 = int(request.form.get('q15'))
        q16 = int(request.form.get('q16'))
        q17 = int(request.form.get('q17'))
        q18 = int(request.form.get('q18'))
        q19 = int(request.form.get('q19'))
        q20 = int(request.form.get('q20'))
        q21 = int(request.form.get('q21'))
        q22 = int(request.form.get('q22'))
        q23 = int(request.form.get('q23'))
        q24 = int(request.form.get('q24'))
        q25 = int(request.form.get('q25'))
        q26 = int(request.form.get('q26'))
        q27 = int(request.form.get('q27'))
        q28 = int(request.form.get('q28'))
        q29 = int(request.form.get('q29'))
        q30 = int(request.form.get('q30'))
        q31 = int(request.form.get('q31'))
        q32 = int(request.form.get('q32'))
        q33 = int(request.form.get('q33'))
        q34 = int(request.form.get('q34'))
        q35 = int(request.form.get('q35'))
        q36 = int(request.form.get('q36'))
        q37 = int(request.form.get('q37'))
        q38 = int(request.form.get('q38'))
        q39 = int(request.form.get('q39'))
        q40 = int(request.form.get('q40'))
        q41 = int(request.form.get('q41'))
        q42 = int(request.form.get('q42'))
        q43 = int(request.form.get('q43'))
        q44 = int(request.form.get('q44'))
        q45 = int(request.form.get('q45'))
        q46 = int(request.form.get('q46'))
        q47 = int(request.form.get('q47'))
        q48 = int(request.form.get('q48'))
        q49 = int(request.form.get('q49'))
        q50 = int(request.form.get('q50'))

        b1 = sum([q1,q2,q3,q4,q5,q6,q7,q8,q9,q10])
        b2 = sum([q11,q12,q13,q14,q15,q16,q17,q18,q19,q20])
        b3 = sum([q21,q22,q23,q24,q25,q26,q27,q28,q29,q30])
        b4 = sum([q31,q32,q33,q34,q35,q36,q37,q38,q39,q40])
        b5 = sum([q41,q42,q43,q44,q45,q46,q47,q48,q49,q50])


        with open(r"/home/ubuntu/static/kmeans.pkl", "rb") as input_file:
            model = pickle.load(input_file)

        y_pred = model.predict(pd.DataFrame({0:[b1],1:[b2],2:[b3],3:[b4],4:[b5]}))
        
        answers = {0:["EXTRAVERSION SCORE: AVERAGE, \nCONSCIENTIOUSNESS SCORE: LOW, \nOPENNESS SCORE: AVERAGE, \nNEUROTICISM SCORE: VERY HIGH, \nAGREEABLENESS SCORE: AVERAGE"],
        1:["EXTRAVERSION SCORE: AVERAGE, \nCONSCIENTIOUSNESS SCORE: LOW, \nOPENNESS SCORE: HIGH, \nNEUROTICISM SCORE: AVERAGE, \nAGREEABLENESS SCORE: AVERAGE"],
        2:["EXTRAVERSION SCORE: VERY LOW, \nCONSCIENTIOUSNESS SCORE: VERY LOW, \nOPENNESS SCORE: VERY LOW, \nNEUROTICISM SCORE: VERY LOW, \nAGREEABLENESS SCORE: VERY LOW"],
        3:["EXTRAVERSION SCORE: HIGH, \nCONSCIENTIOUSNESS SCORE: AVERAGE, \nOPENNESS SCORE: HIGH, \nNEUROTICISM SCORE: VERY HIGH, \nAGREEABLENESS SCORE: AVERAGE"],
        4:["EXTRAVERSION SCORE: HIGH, \nCONSCIENTIOUSNESS SCORE: AVERAGE, \nOPENNESS SCORE: AVERAGE, \nNEUROTICISM SCORE: HIGH, \nAGREEABLENESS SCORE: HIGH"]}
        
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE candidate SET personality_score = '{}' WHERE can_id = 11113".format(answers[y_pred[0]][0]))
        mysql.connection.commit()
        cursor.close()
    return render_template('dashboard.html')

@app.route('/dashboard', methods =["GET", "POST"])
def dashboard():
    return render_template('dashboard.html')

@app.route('/job-listing', methods =["GET", "POST"])
def job_listing():
    return render_template('job-listing.html')

@app.route('/employers-listing', methods =["GET", "POST"])
def employers_listing():
    return render_template('employers-listing.html')

@app.route('/candidates-dashboard-all-job', methods =["GET", "POST"])
def candidates_dashboard_all_job():

    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT * FROM jobs ''')
    data=cursor.fetchall()
    mysql.connection.commit()
    return render_template('candidates-dashboard-all-job.html', job_data = data)


@app.route('/candidates-dashboard-applied-job', methods =["GET", "POST"])
def candidates_dashboard_applied_job():
    return render_template('candidates-dashboard-applied-job.html')

@app.route('/candidates-dashboard-recommended-job', methods =["GET", "POST"])
def candidates_dashboard_recommended_job():

    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT * FROM jobs where job_name="Business Analyst"''')
    data=cursor.fetchall()
    mysql.connection.commit()
    return render_template('candidates-dashboard-recommended-job.html', job_data = data)

@app.route('/candidates-dashboard-my-profile', methods =["GET", "POST"])
def candidates_dashboard_my_profile():
    return render_template('candidates-dashboard-my-profile.html')

@app.route('/candidates-dashboard-my-profile-send', methods =["GET", "POST"])
def candidates_dashboard_my_profile_send():

    if request.method == 'GET':
        return "Couldn't reach to the server"
     
    if request.method == 'POST':
        candidate_name = request.form.get('candidate_name')
        location = request.form.get('location')
        email_address = request.form.get('email_address')
        phone_country_code = request.form.get('phone_country_code')
        phone = request.form.get('phone')
        date_of_birth = request.form.get('date_of_birth')
        candidate_qualification = request.form.get('candidate_qualification')
        candidate_availability =  request.form.get('candidate_availability')
        experience = int(request.form.get('experience'))
        about_candidate = request.form.get('about_candidate')

        with open(r"/home/ubuntu/encoder_model.pkl", "rb") as input_file:
            encoder = pickle.load(input_file)

        new_model = tf.keras.models.load_model('/home/ubuntu/static/cnn_model')

        encoded_docs = [one_hot(about_candidate, 1000)]
        padded_text = pad_sequences(encoded_docs, maxlen=1000, padding='post')
        prediction = new_model.predict(padded_text)
        result = encoder.inverse_transform(prediction)
        model_prediction = result[0]

        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO candidate (can_about,can_availability,can_dob,can_email,can_experience,can_location,can_name,can_phone,can_phone_ctry_code,can_qualification,recommended_jobs) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(about_candidate, candidate_availability, date_of_birth, email_address, experience, location, candidate_name, phone, phone_country_code, candidate_qualification, model_prediction))
        mysql.connection.commit()
        cursor.close()
        val = "Details Entered Successfully" + candidate_name

    return render_template('dashboard.html')

@app.route('/job-details?<job_id>', methods =["GET", "POST"])
def job_details(job_id):
    print(job_id)

    cursor = mysql.connection.cursor()
    query = "SELECT * FROM jobs where job_id='" + job_id + "'"
    cursor.execute(query)
    data=cursor.fetchall()
    mysql.connection.commit()

    return render_template('job-details.html', job_name = data[0][1],
    job_category = data[0][2], job_pub_on = data[0][3], job_vacant_count = data[0][4], job_type = data[0][5], job_experience = data[0][6], 
    job_due_date = data[0][7], job_location = data[0][8], job_skills = data[0][9], job_is_remote = data[0][10], job_desc = data[0][11],
    job_company_name = data[0][12], job_salary = data[0][13])

@app.route('/candidate-details?<candidate_id>', methods =["GET", "POST"])
def candidate_details(candidate_id):
    cursor = mysql.connection.cursor()
    query = "SELECT * FROM candidate where can_id='" + candidate_id + "'"
    cursor.execute(query)
    data=cursor.fetchall()
    mysql.connection.commit()

    return render_template('candidate-details.html', can_name = data[0][1],
    can_dob = data[0][2], can_email = data[0][3], can_phone = data[0][4], can_phone_ctry_code = data[0][5], can_experience = data[0][6],  can_about = data[0][7], 
    can_availability = data[0][9], can_location = data[0][10], can_qualification = data[0][11], recommended_jobs = data[0][12], logical_score = data[0][13],
    personality_score = data[0][14])

@app.route('/web-form', methods =["GET", "POST"])
def web_form():
    return render_template('web_form.html')

@app.route('/web-form-send', methods =["GET", "POST"])
def web_form_send():

    if request.method == 'GET':
        return "Couldn't reach to the server"
     
    if request.method == 'POST':
        candidate_name = request.form.get('candidate_name')
        location = request.form.get('location')
        email_address = request.form.get('email_address')
        phone_country_code = request.form.get('phone_country_code')
        phone = request.form.get('phone')
        date_of_birth = request.form.get('date_of_birth')
        candidate_qualification = request.form.get('candidate_qualification')
        candidate_availability =  request.form.get('candidate_availability')
        experience = int(request.form.get('experience'))
        about_candidate = request.form.get('about_candidate')
        input = request.form
        resume = request.files['resume']
        resume.save(os.path.join('/home/ubuntu/static/resume', secure_filename(resume.filename)))

        with open(r"/home/ubuntu/encoder_model.pkl", "rb") as input_file:
            encoder = pickle.load(input_file)

        new_model = tf.keras.models.load_model('/home/ubuntu/static/cnn_model')

        encoded_docs = [one_hot(about_candidate, 1000)]
        padded_text = pad_sequences(encoded_docs, maxlen=1000, padding='post')
        prediction = new_model.predict(padded_text)
        result = encoder.inverse_transform(prediction)
        model_prediction = result[0]

        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO candidate (can_about,can_availability,can_dob,can_email,can_experience,can_location,can_name,can_phone,can_phone_ctry_code,can_qualification,recommended_jobs) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(about_candidate, candidate_availability, date_of_birth, email_address, experience, location, candidate_name, phone, phone_country_code, candidate_qualification, model_prediction))
        mysql.connection.commit()
        cursor.close()
        val = "Details Entered Successfully" + candidate_name

    return render_template('dashboard.html')


    
@app.route('/employer', methods =["GET", "POST"])
def employer():
    return render_template('employer.html')

@app.route('/post-a-job', methods =["GET", "POST"])
def postajob():
    return render_template('post-job.html')

@app.route('/post-a-job-send', methods =["GET", "POST"])
def postajobsend():
     
    if request.method == 'POST':
        job_name = request.form.get('job_name')
        job_category = request.form.get('job_category')
        job_pub_on = datetime.today().strftime('%d-%m-%Y')
        job_vacant_count = request.form.get('job-vacant-count')
        job_type = request.form.get('job_type')
        job_experience = request.form.get('exp-required')
        job_due_date = request.form.get('job_due_date')
        job_location = request.form.get('job-location')
        job_skills = request.form.get('skills-req')
        job_is_remote = request.form.get('is-remote')
        job_desc = request.form.get('job_desc')
        job_company_name = request.form.get('job_company_name')
        job_salary = request.form.get('job_salary')
        cursor = mysql.connection.cursor()
        print(job_name,job_category,job_pub_on,job_vacant_count, job_type,job_experience, job_due_date, job_location, job_skills, job_is_remote,job_desc, job_company_name, job_salary)
        cursor.execute(''' INSERT INTO jobs (job_name,job_category,job_pub_on,job_vacant_count, job_type,job_experience, job_due_date, job_location, job_skills, job_is_remote,job_desc, job_company_name, job_salary) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(job_name,job_category,job_pub_on,job_vacant_count, job_type,job_experience, job_due_date, job_location, job_skills, job_is_remote,job_desc, job_company_name, job_salary))
        mysql.connection.commit()
        cursor.close()
    return render_template('employer.html')
    
@app.route('/employer-manage-job', methods =["GET", "POST"])
def employer_manage_job():
    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT * FROM jobs ''')
    data=cursor.fetchall()
    mysql.connection.commit()

    # candidate_url = "url_for('candidate-approve' ,candidate_id=" + str(data[0]) + ")"

    return render_template('employer-manage-job.html', job_data = data)

@app.route('/resume-manage', methods =["GET", "POST"])
def resume_manage():

    def convert_pdf_to_string(file_path):

        output_string = StringIO()
        with open(file_path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)

        return(output_string.getvalue())

    mypath='/static/resume' 
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    def create_profile(file):
        text = convert_pdf_to_string(file) 
        text = str(text)
        text = text.replace("\\n", "")
        text = text.lower()
        
        keyword_dict = pd.read_csv('/home/ubuntu/static/New_Keywords.csv')
        SM_words = [nlp(text) for text in keyword_dict['Sales & Marketing'].dropna(axis = 0)]
        LL_words = [nlp(text) for text in keyword_dict['Legal'].dropna(axis = 0)]
        LE_words = [nlp(text) for text in keyword_dict['Law Enforcement'].dropna(axis = 0)]
        IT_words = [nlp(text) for text in keyword_dict['Information Technology'].dropna(axis = 0)]
        HR_words = [nlp(text) for text in keyword_dict['Human Resources'].dropna(axis = 0)]
        HS_words = [nlp(text) for text in keyword_dict['Healthcare & Human Services'].dropna(axis = 0)]
        FH_words = [nlp(text) for text in keyword_dict['Food Service & Hospitality'].dropna(axis = 0)]
        ES_words = [nlp(text) for text in keyword_dict['Engineering & Scientific'].dropna(axis = 0)]
        EL_words = [nlp(text) for text in keyword_dict['Education & Learning'].dropna(axis = 0)]
        CC_words = [nlp(text) for text in keyword_dict['Creative & Cultural'].dropna(axis = 0)]
        AF_words = [nlp(text) for text in keyword_dict['Accounting & Finance'].dropna(axis = 0)]

        matcher = PhraseMatcher(nlp.vocab)
        matcher.add('Sales&Marketing', None, *SM_words)
        matcher.add('Legal', None, *LL_words)
        matcher.add('LawEnforcement', None, *LE_words)
        matcher.add('InformationTechnology', None, *IT_words)
        matcher.add('HumanResources', None, *HR_words)
        matcher.add('Healthcare&HumanServices', None, *HS_words)
        matcher.add('FoodService&Hospitality', None, *FH_words)
        matcher.add('Engineering&Scientific', None, *ES_words)
        matcher.add('Education&Learning', None, *EL_words)
        matcher.add('Creative&Cultural', None, *CC_words)
        matcher.add('Accounting&Finance', None, *AF_words)
        doc = nlp(text)
        
        d = []  
        matches = matcher(doc)
        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id] 
            span = doc[start : end]
            d.append((rule_id, span.text))      
        keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
        
        df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
        df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
        df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
        df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
        df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
        
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        
        name = filename.split('_')
        name2 = name[0]
        name2 = name2.lower()
        name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
        
        dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
        dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

        return(dataf)

    final_database=pd.DataFrame()
    i = 0 
    while i < len(onlyfiles):
        file = onlyfiles[i]
        dat = create_profile(file)
        final_database = final_database.append(dat)
        i +=1

    final_database = final_database[final_database['Count'].notna()]
    final_database.sort_values('Count', ascending=False)

    print(final_database)
    return render_template('resume-manage.html', resume_data = final_database)

@app.route('/job-delete?<job_id>', methods =["GET", "POST"])
def job_delete(job_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM jobs WHERE job_id = " + job_id + "; ")
    # data=cursor.fetchall()
    mysql.connection.commit()
    return render_template('employer-manage-job.html')

@app.route('/employer-all-applicants', methods =["GET", "POST"])
def employer_all_applicants():
    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT * FROM candidate ''')
    data=cursor.fetchall()
    mysql.connection.commit()

    # candidate_url = "url_for('candidate-approve' ,candidate_id=" + str(data[0]) + ")"

    return render_template('employer-all-applicants.html', candidate_data = data)

@app.route('/candidate-approve?<candidate_id>', methods =["GET", "POST"])
def candidate_approve(candidate_id):
    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE candidate SET selected = 1 WHERE can_id = " + candidate_id + "; ")
    # data=cursor.fetchall()
    mysql.connection.commit()
    return render_template('employer-all-applicants.html')

@app.route('/candidate-reject?<candidate_id>', methods =["GET", "POST"])
def candidate_reject(candidate_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM candidate WHERE can_id = " + candidate_id + "; ")
    # data=cursor.fetchall()
    mysql.connection.commit()

    return render_template('employer-all-applicants.html')

@app.route('/candidate-delete?<candidate_id>', methods =["GET", "POST"])
def candidate_delete(candidate_id):
    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE candidate SET selected = 0 WHERE can_id = " + candidate_id + "; ")
    # data=cursor.fetchall()
    mysql.connection.commit()

    return render_template('employer-all-applicants.html')

@app.route('/employer-shortlisted-candidates', methods =["GET", "POST"])
def employer_shortlisted_candidates():
    cursor = mysql.connection.cursor()
    cursor.execute(''' SELECT * FROM candidate where selected = 1''')
    data=cursor.fetchall()
    mysql.connection.commit()
    return render_template('employer-shortlisted-candidates.html', candidate_data = data)



@app.route('/login', methods =["GET", "POST"])
def form():
    return render_template('log-in-register.html')

@app.route('/login-landing', methods =["GET", "POST"])
def login_landing():
    return render_template('login_landing.html')
 
def _convert(tup, di):
    di = dict(tup)
    return di

@app.route('/login_send', methods = ['GET', 'POST'])
def login():
    if request.method == 'GET':
        return "Login via the login Form"
     
    if request.method == 'POST':
        user_name = request.form.get('user_name')
        password = request.form.get('password')
        email_id = request.form.get('email')
        position = int(request.form.get('position'))
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO login_signup VALUES(%s,%s,%s,%s)''',(password,user_name,email_id,position))
        mysql.connection.commit()
        cursor.close()
        # val = "Details Entered Successfully" + user_name
        return render_template('login_landing.html')

@app.route('/login_receive', methods = ['GET', 'POST'])
def loginandsignup():
    if request.method == 'GET':
        return "Login via the login Form"
     
    if request.method == 'POST':
        password = request.form.get('password')
        email_id = request.form.get('email')
        cursor = mysql.connection.cursor()
        cursor.execute(''' SELECT email_id, password FROM login_signup ''')
        data=cursor.fetchall()
        mysql.connection.commit()

        my_dict={}
        f_data = _convert(data,my_dict)

        cursor.execute(''' SELECT email_id, position FROM login_signup ''')
        data=cursor.fetchall()
        mysql.connection.commit()

        my_dict_2={}
        f_data_2 = _convert(data,my_dict_2)

        if email_id in f_data.keys():
            if f_data[email_id] == password:
                if f_data_2[email_id] == 1:
                    return render_template('dashboard.html')
                else:
                    return render_template('employer.html')
        return f"Error 404"

if __name__=='__main__':
   app.run(host='0.0.0.0',port=8080)

