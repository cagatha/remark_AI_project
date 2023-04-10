import os
import psycopg2
import requests
import multiprocessing
import time
import re
import time
import pandas as pd
import numpy as np
import json
import jieba.analyse
import jieba
import string
import warnings

from flask_sqlalchemy import SQLAlchemy
from ast import literal_eval
from flask_bootstrap import Bootstrap
from selenium import webdriver
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_session import Session




warnings.filterwarnings('ignore')

app =Flask(__name__)

app.secret_key = "my_secret_key"

bootstrap = Bootstrap(app)

conn = psycopg2.connect(
    host='ec2-34-236-103-63.compute-1.amazonaws.com',
    database='d6bqsucsud4mkl',
    user='jrxovyrmkvtxkw',
    password='15371b38bb3ae2fccb27f1e4d69312db8c26bb7fb343e14b0a05510d6b342a82'
    ,
)

cursor = conn.cursor()

# login_manager = LoginManager()
# login_manager.init_app(app)

# Initialize the Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
chrome_options.add_argument("--headless") #無頭模式
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")

activity_type_names = {
    "0": '其他',
    "1": '美食',
    "2": '景點',
    "3": '住宿'
}

def connect_to_database():
    conn = psycopg2.connect(
        host='ec2-34-236-103-63.compute-1.amazonaws.com',
        database='d6bqsucsud4mkl',
        user='jrxovyrmkvtxkw',
        password='15371b38bb3ae2fccb27f1e4d69312db8c26bb7fb343e14b0a05510d6b342a82'
        ,
    )
    return conn

@app.route('/testdb')
def testdb():
    conn = connect_to_database()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO example_table (name) VALUES ('Peter')")
    conn.commit()
    cursor.close()
    conn.close()
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index')
def index_calendar():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def do_login():
    # 獲取使用者輸入的帳號和密碼
    username = request.form['username']
    password = request.form['password']

    # 連接到 PostgreSQL 資料庫
    conn = connect_to_database()

    # 執行 SQL 查詢
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM public.user WHERE account = %s AND password = %s",(username, password))
    user = cursor.fetchone()
    cursor.execute("SELECT * FROM public.user WHERE account = %s ", (username,))
    oneuser = cursor.fetchone()
    cursor.close()
    

 
    #如果users沒有在database裡面轉到註冊頁面
    if not oneuser:
        return redirect('/register')

    # 驗證使用者帳號和密碼s
    if user is not None:
        user_id=user[0]
        session['user_id']=user_id
        # 登入成功，導向使用者主頁面
        return render_template('index.html', username=username)
        # return render_template('index.html')
    else:
        # 登入失敗，顯示錯誤訊息
        error = '帳號或密碼錯誤'
        return render_template('login.html', error=error)
        # return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    connect_to_database()
    cursor = conn.cursor()
    if request.method == 'POST':
        account = request.form['username']
        password = request.form['password']
        name = request.form['name']
        birthday = request.form['birthday']
        email = request.form['email']
        gender = request.form['gender']
        status = request.form['status']
        nickname = request.form['nickname']
        cursor.execute('INSERT INTO "user" (account, password, name, birthday, email, gender, status, nickname) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)', (account, password, name, birthday, email, gender, status, nickname))
        conn.commit()
        cursor.close()

        return redirect('/login')
    return render_template('register.html')

@app.route('/friend')
def friend():
    connect_to_database()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM public.\"user\"")
    friends = cursor.fetchall()
    cursor.close()
    return render_template('friend_test.html', friends=friends)

@app.route('/api/friend')
def api_friend():
    cursor = conn.cursor()
    # cursor.execute("SELECT * FROM public.\"user\" WHERE user_id IN (SELECT friend_id FROM friend WHERE user_id = 1)")
    user_id = session.get("user_id")
    if 'user_id' in session:
        cursor.execute("SELECT * FROM public.\"user\" WHERE user_id IN (SELECT friend_id FROM friend WHERE user_id = %s)", (user_id,))
        friend_data = cursor.fetchall()
        cursor.close()
    
        friend_list = []
        for friend in friend_data:
            friend_dict = {'name': friend[3], 'nickname': friend[13], 'email': friend[5], 'gender': friend[6]}
            friend_list.append(friend_dict)
    
        return jsonify({'friend': friend_list})
    else:
        "userid not in session"

@app.route('/add_friend', methods=['POST'])
def add_friend():
    # 從 POST 請求中獲取用戶帳號或用戶名稱
    input_val = request.form.get('input')
    # 設定 session['user_id'] 的值
    

    # 如果使用了用戶帳號，從 user 資料庫中查詢用戶
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM public.user WHERE account = %s", (input_val,))
        result = cursor.fetchone()

        # 如果用戶帳號查詢到用戶，插入用戶到 friend 資料庫
        if result:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO friend (user_id, friend_id) VALUES (%s, %s)", (session['user_id'], result[0]))
                cursor.execute("INSERT INTO friend (user_id, friend_id) VALUES (%s, %s)", (result[0], session['user_id']))
            conn.commit()
        else:
            # 如果用戶帳號無效，則從 user 資料庫中查詢用戶名稱
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM public.user WHERE name = %s", (input_val,))
                result = cursor.fetchone()

                # 如果用戶名稱查詢到用戶，插入用戶到 friend 資料庫
                if result:
                    with conn.cursor() as cursor:
                        cursor.execute("INSERT INTO friend (user_id, friend_id) VALUES (%s, %s)", (session['user_id'], result[0]))
                        cursor.execute("INSERT INTO friend (user_id, friend_id) VALUES (%s, %s)", (result[0], session['user_id']))
                    conn.commit()
                else:
                    # 如果都查詢不到用戶，則顯示錯誤提示
                    flash('用戶帳號或用戶名稱無效')
                    return redirect('/friend')

    return redirect('/friend')


@app.route('/calendar', methods = ['POST', 'GET'])
def calendar():
    conn = connect_to_database()
    cursor = conn.cursor()
    # user_id = session.get("user_id")    
    cursor.execute("select * from calendar where founder_id = %s", (1,))
    rows = cursor.fetchall()
    cursor.close()

    while not rows:
        cursor.execute("select * from calendar where founder_id = %s", (1,))
        rows = cursor.fetchall()
    return render_template('calendar_test.html', rows=rows)
    # else:
    #     return "userd not in session"


@app.route('/remark', methods = ['POST', 'GET'])
def remark():
    connect_to_database()
    cursor = conn.cursor()
    cursor.execute("select *from calendar where calendar_id=('%d')" %(1))  
    rows = cursor.fetchall()
    

    while not rows:
        cursor.execute("select *from calendar where calendar_id=('%d')" %(1)) 
        rows = cursor.fetchall()
    return render_template('remark1_look.html', activities=rows,activity_type_names=activity_type_names)


@app.route('/remark2')
def home():
    return render_template('new_test5.html')

@app.route('/search', methods=['POST'])
def search():
	search_term = request.form['search']
	results = combine(search_term)
    # results=session['results']
	return render_template('new_test5.html', results=results)

def combine(search_term):
    
    def yahootravel(search_term):
        returnNumber = ""
        search_results=[]
        my_headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
        browser=webdriver.Chrome(r'C:/Users/agath/Desktop/flask/chromedriver.exe',options=chrome_options)
        browser.get("https://travel.yahoo.com.tw/")
        WebDriverWait(browser,30).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR,"div#SearchForm")))
        element=browser.find_element(By.CSS_SELECTOR,"div#SearchForm input")
        browser.find_element(By.CSS_SELECTOR,"div#SearchForm input").click()
        element.send_keys(search_term)
        submit=browser.find_element(By.CSS_SELECTOR,'button.SiteSearchBTN').click()
        element=WebDriverWait(browser, 30).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR,"body.search-page")))
        # titleList=[]
        # image_url=[]
        # description=[]
        results1=[]
        productcount=5
        while True:   
            browser.find_element(By.TAG_NAME,'body').send_keys(Keys.END) 
            soup=BeautifulSoup(browser.page_source,"html.parser")
            titles=soup.select("div.item_txt div.item_topic.dotdotdot")
            links=soup.select("div.MainList div.mask div.item_block a")
            pics=soup.select("div.MainList div.item_block div.item_img")
        
            if len(pics)>=productcount: #這邊讓prices到50就不要在往下拉了
                break
        for index,(title,link,pic) in enumerate(zip(titles,links,pics)):
            titleDic=title.text
            #returnNumber = returnNumber + title.text + "<br>"
            linkDic=("https://travel.yahoo.com.tw/"+link.get("href"))
            #print("https://travel.yahoo.com.tw/"+link.get("href"))
            # returnNumber = returnNumber + "https://travel.yahoo.com.tw/"+link.get("href") + "<br>"
            #print("https://"+pic.get("style").split("url(https:")[1].split("https://")[1].split(")")[0])
            image_urlDic=("https://"+pic.get("style").split("url(https:")[1].split("https://")[1].split(")")[0])
            # returnNumber = returnNumber + "https://"+pic.get("style").split("url(https:")[1].split("https://")[1].split(")")[0]+"<br>"
            
            results1.append({"title": titleDic, "link": linkDic, "image_url": image_urlDic})
            # print(stories)
            if index==productcount-1:#有可能會超過50 在設定條件只print出50條
                break
        return results1
        # print(results1)
        browser.close()
        
    
    
    def pixnet(search_term):
        try:
            my_headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}
            
            
            browser=webdriver.Chrome(r'C:/Users/agath/Desktop/final project/chromedriver.exe',options=chrome_options)
            browser.get("https://www.pixnet.net/blog/articles/group/2")
            
            WebDriverWait(browser,15).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR,"span.pixnavbar__top-bar__icon-item.pixnavbar__top-bar__icon-tag-search")))
            c=browser.find_element(By.CSS_SELECTOR,"span.pixnavbar__top-bar__icon-item.pixnavbar__top-bar__icon-tag-search").click()
            element=browser.find_element(By.CSS_SELECTOR,'input.pixnavbar__tag-search-drawer__search-input')
            element.send_keys(search_term)
            submit=browser.find_element(By.CSS_SELECTOR,'button.pixnavbar__tag-search-drawer__search-btn.has-value').click()
            
            
            element=WebDriverWait(browser, 15).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR,"div.sdysz0-0.flSPjY.sc-1d2bccd-3.ilQPzG img.sc-1d2bccd-4.jDsxtU")))
            
            results2=[]
            productcount=5
            while True:  
                browser.find_element(By.TAG_NAME,'body').send_keys(Keys.END) 
                time.sleep(0.2)
                element=WebDriverWait(browser, 15).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR,"div.sdysz0-0.flSPjY.sc-1d2bccd-3.ilQPzG img.sc-1d2bccd-4.jDsxtU")))
                
                soup=BeautifulSoup(browser.page_source,"html.parser")
                titles=soup.select("h3.sc-1ym3hzi-0 a.sc-1ym3hzi-1.eKtpkt")
                links=soup.select("h3.sc-1ym3hzi-0 a.sc-1ym3hzi-1.eKtpkt")   
                pictures=soup.select("div.kl4nvp-0.jnGZXj div.sc-1d2bccd-0 div.sdysz0-0 img")
                if len(pictures)>=productcount:
                    break
            for index,(title,link,pic) in enumerate(zip(titles,links,pictures)):        
                # print(str(index+1)+":"+title.text[:40]+"...")
                titleDic=(title.text[:40]+"...")
                # print(link.get("href"))
                linkDic=link.get("href")
                # print(pic.get("src"))
                image_urlDic=pic.get("src")
                results2.append({"title": titleDic, "link": linkDic, "image_url": image_urlDic})
                if index==productcount-1:
                    break
            return results2
            # print(results2)
            browser.close()
        except:
            pass
    
    try:
        output1 = pixnet(search_term)
    except:
        output1 = []
    
    output2 = yahootravel(search_term)

    if output1:
        results = output2 + output1
    else:
        results = output2
        
    session["results"]=results    
    return results
    
            
    if __name__ == '__main__':
        # Create two processes for the two functions
        # browser=webdriver.Chrome(r'C:/Users/agath/Desktop/final project/chromedriver.exe')
        p1 = multiprocessing.Process(target=yahootravel(search_term))
        p2 = multiprocessing.Process(target=pixnet(search_term))
        
        # Start the processes
        p1.start()
        p2.start()
        
        # Wait for the processes to complete
        p1.join()
        p2.join()
        
        # browser.close()    


    
@app.route('/save', methods=['POST'])
def save():
    connect_to_database()
    cursor = conn.cursor()
    selected_items = request.form.getlist('selected_items')
    action = request.form.get('action')
    if action == 'no-save':
        return redirect(url_for('showai')) 
    else:  
        all_items=session['results']               
        for result in all_items:
            if result['link'] in selected_items:                
                cursor.execute("INSERT INTO remark (remark_name,remark_image,remark_url,calendar_id,founder_id) VALUES ('%s','%s','%s','%d','%d')" %(result['title'],result['image_url'],result['link'],1,23))                               
                conn.commit()

    return redirect(url_for('showai')) 


@app.route('/showai', methods=['GET'])
def showai():
    recommendList = filtering1()
    similar_items_list = filtering2()
    session['recommendList'] = recommendList
    session['similar_items_list'] = similar_items_list
    return render_template('ai_temp_look.html', recommendList=recommendList, similar_items_list=similar_items_list)

def filtering1():
    connect_to_database()
    cursor = conn.cursor()
    # df = pd.read_csv(r'C:/Users/agath/Desktop/final project/traveltest.csv',encoding='utf-8')
    df = pd.read_sql("""
        SELECT * from remark;
    """, conn)
    
    df['title'] = df['remark_name']
    df['link'] = df['remark_url']
    df['image_url'] = df['remark_image']
    
    # 處理字串問題
    printable = set(string.printable)
    
    # remove any non-printable characters from the given string
    def remove_nonprintable(s):
        return ''.join(filter(lambda x: x in printable, s))
           
    # Initialize empty column
    df['plotwords'] = ''
    # function to get keywords from a text
    #這邊是用 td-idf from jieba, TCSP mandarin read_stopwords沒有用
    def get_keywords(x):
        plot = x
        
        # extract keywords using TF-IDF
        keywords = jieba.analyse.extract_tags(plot, topK=10, withWeight=False, allowPOS=('n', 'v', 'a'))
        
        # return list of keywords
        return keywords
    
    #新的column "plotwords"是title新抓出來的keywords
    df['plotwords'] = df['title'].apply(get_keywords)
        
    #新增一個dataframe叫做df_keys 只保留了title 跟把keywords放進去
    #原始的plotwords是一個list,這邊用bag_words function把它們分開
    #這邊只留下兩個columns "title" 跟"keyword"看之後要不要分開
    df_keys = pd.DataFrame() 
    df_keys['title'] = df['title']
    df_keys['keywords'] = ''
    
    def bag_words(x):
        return(' '.join(x['plotwords']))
    df_keys['keywords'] = df.apply(bag_words, axis = 1)
    
         
    # Define custom tokenizer function using jieba
    def tokenize(text):
        # Tokenize the text using jieba
        tokens = jieba.cut(text)
        # Join the tokens back into a single string
        return " ".join(tokens)
    
    # Instantiate CountVectorizer with custom tokenizer
    cv = CountVectorizer(tokenizer=tokenize)
    
    cv_mx = cv.fit_transform(df_keys['keywords'])
    
    # create cosine similarity matrix
    cosine_sim = cosine_similarity(cv_mx, cv_mx)
       
    # create list of indices for later matching
    df_keys['title'] = df_keys['title'].str.encode('utf-8').str.decode('utf-8')
    df_keys['keywords'] = df_keys['keywords'].str.encode('utf-8').str.decode('utf-8')
    
    indices = pd.Series(df_keys.index, index=df_keys['title'])   
    indices = pd.Series(range(len(df_keys['title'])), index=df_keys['title'])
    
    #import founderid 
    #founderid should be able to import directly這邊先寫死
    #select the last saved title
    cursor.execute("select *from remark where founder_id=('%d') order by remark_id desc limit 1" %(11))  
    rows = cursor.fetchall()
    
    while not rows:
        rows = cursor.fetchall()
    likedTitle=rows[0][3]

    
    
    def recommend_link(title=likedTitle, n=5, cosine_sim = cosine_sim):
        # find the index label(s) that contain the substring 'title'
        matching_indices = indices[indices.index.str.contains(f"^{re.escape(likedTitle)}$")]
        
        # retrieve matching title index
        if len(matching_indices) == 0:
            return
        else:
            idx = matching_indices.values[0]
            
            # cosine similarity scores of movies in descending order
            scores = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
            
            # top n most similar titles indexes
            # use 1:n because 0 is the same title entered
            top_n_idx = list(scores.iloc[1:n+1].index)    
            # remove the title that was used for matching
            top_n_idx = [i for i in top_n_idx if df_keys.iloc[i]['title'] != title]
                   
            recommend=(df[['title','link','image_url']].iloc[top_n_idx])
            recommend = recommend.drop_duplicates(subset=['title'])
            recommend_list = recommend[['title', 'link', 'image_url']].to_dict(orient='records')
        return recommend_list

    recommendList = recommend_link()
    # print(recommendList)
    return recommendList

def filtering2():
    connect_to_database()
    cursor = conn.cursor()
    df = pd.read_sql("""
        SELECT t1."user_id", t1.birthday, t1.gender, t1.status, t1.age, t2.founder_id, t2.remark_name, t2.remark_url, t2.remark_image
        FROM "user" AS t1 
        JOIN remark AS t2 ON t1."user_id" = t2.founder_id;
    """, conn)
    
    # close database connection
    # conn.close()
    
    #keep the columns needed    
       
    df['title'] = df['remark_name']
    df['link'] = df['remark_url']
    df['image_url'] = df['remark_image']
    df = df.drop(['remark_name', 'remark_url','remark_image','birthday'], axis=1)
    df = df.dropna()

    # Convert categorical variables into numeric form using one-hot encoding
    data = pd.get_dummies(df, columns=['gender', 'status'])

    # Create a label encoder object
    le = LabelEncoder()

    # Encode the 'title' column
    data['title_encoded'] = le.fit_transform(data['title'])
    data2=data[['user_id', 'title_encoded','gender_female','gender_male','status_married','status_single','age']]


    data2['count'] = 1
    # grouped_data = data.groupby(['user_id', 'title'], as_index=False).count()

    # Group the data by user_id and title columns
    grouped_data = data2.groupby(['user_id', 'title_encoded'], as_index=False).count()


    # Create a user-item interaction matrix
    matrix = grouped_data.pivot_table(index='user_id', columns='title_encoded', values='count', fill_value=0)

    # Create user profiles based on demographic information
    # This gives us the average value of each one-hot encoded feature for each user.
    data3 = data2.drop(['title_encoded'], axis=1)
    user_profiles = data3.groupby('user_id').mean()



    # Calculate the similarity between user profiles
    similarity_matrix_dem = cosine_similarity(user_profiles)

    similarity_matrix_dem = pd.DataFrame(similarity_matrix_dem, index=user_profiles.index, columns=user_profiles.index)

    # Find similar users based on both collaborative filtering and demographic filtering
    # user_id will need to be imported 這邊先寫死
    user_id = 1
    similar_users_cf = matrix.corrwith(matrix.loc[user_id], axis=1).sort_values(ascending=False)[:5].index
    similar_users_dem = similarity_matrix_dem[user_id].argsort()[:-5-1:-1]
    similar_users_dem_ids = similarity_matrix_dem.index[similar_users_dem]
    similar_users = list(set(similar_users_dem_ids) & set(similar_users_cf))
    similar_users.remove(user_id)

    # Extract saved items of similar users
    similar_items = matrix.loc[similar_users].sum().sort_values(ascending=False)[:5]

    # Find items that user_id has already liked
    items_liked = matrix.loc[user_id][matrix.loc[user_id] > 0].index.tolist()

    # Remove items that user_id has already liked from similar_items
    similar_items = similar_items.drop(items_liked, errors='ignore')

    #merge it back to the original df with left join, and only keep the records when count=1
    item_data = data[['title', 'link','image_url','title_encoded']]
    similar_items = pd.DataFrame(similar_items, columns=['count'])

    similar_items = similar_items[similar_items['count'] == 1]



    #take out the duplicates since we merge on titles
    similar_items = similar_items.merge(item_data, on='title_encoded', how='left')
    similar_items = similar_items.drop_duplicates(subset=['title'])



    #convert into a list of dictionaries for later use
    similar_items_list = similar_items[['title', 'link', 'image_url']].to_dict(orient='records')
    return similar_items_list
    
@app.route('/saveai', methods=['POST'])
def saveai():
    connect_to_database()
    cursor = conn.cursor()
    selected_items = request.form.getlist('selected_items')
    action = request.form.get('action')

    if action == 'no-save':
        return redirect(url_for('remark3')) 
    else:  
        recommendList = session.get('recommendList', [])  # get the value for 'recommendList' key, or an empty list if it doesn't exist
        similar_items_list = session.get('similar_items_list', [])  # get the value for 'similar_items_list' key, or an empty list if it doesn't exist
        all_items = recommendList + similar_items_list  # concatenate the two lists
        for result in all_items:
            if result['link'] in selected_items:
                cursor.execute("INSERT INTO remark (remark_name,remark_image,remark_url,calendar_id,founder_id) VALUES ('%s','%s','%s','%d','%d')" %(result['title'],result['image_url'],result['link'],1,23))
                conn.commit()
        cursor.close()

    return redirect(url_for('remark3')) 

@app.route('/remark3', methods = ['POST', 'GET'])
def remark3():
    connect_to_database()
    cursor = conn.cursor()
    cursor.execute("select *from remark where calendar_id=('%d')" %(1))  
    results = cursor.fetchall()
    cursor.execute("select *from calendar where calendar_id=('%d')" %(1))  
    rows = cursor.fetchall() 
    while not results or not rows:
        time.sleep(1)  # wait for 1 second before re-querying the database
        cursor.execute("select * from remark where calendar_id = (%s)", (1,))
        results = cursor.fetchall()
        cursor.execute("select * from calendar where calendar_id = (%s)", (1,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    return render_template('remark3_look.html', results=results,activities=rows,activity_type_names=activity_type_names)


if __name__ == '__main__':
    #app.run(debug = True)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0',port=port)