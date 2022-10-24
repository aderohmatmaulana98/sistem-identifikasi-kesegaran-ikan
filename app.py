from MySQLdb.cursors import Cursor
from flask import Flask, app, json, render_template, session, redirect, url_for, request
from flask_mysqldb import MySQL, MySQLdb
import os
import cv2
import math
import numpy as np
from numpy.lib.function_base import select
import pandas as pd
from scipy.sparse import data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import sqlalchemy
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "iniSecretKeyKu2019"
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_ikan'
ALLOWED_EXTENSION = set(['png', 'jpeg', 'jpg'])
app.config['UPLOAD_FOLDER'] = 'uploads'
mysql = MySQL(app)
cnx = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3306/db_ikan')

knn_jarak = []
knn_jarak1 = []
class KNN :
    def __init__(self, k=3):
        self.K=k
    def train(self, X, y):
        self.X_train = X
        self.y_train =y
        
    def predict(self, X):
        y_prediksi = [self._prediksi(x) for x in X]
        return np.array(y_prediksi)
    
    def _prediksi(self,x):
        #Hitung Jarak ke semua data training
        jarak_titik = [self.jarak(x,x_train) for x_train in self.X_train] 
        print(jarak_titik)
        if knn_jarak == [] :
            knn_jarak.append(jarak_titik)
        else :
            knn_jarak[0:]=[jarak_titik]
        # urutkan berdasarkan jarak terdekat, ambil sejumlah K
        k_terbaik = np.argsort(jarak_titik)[0:int(self.K)]
        print(k_terbaik)
        #Ambil label k_terbaik
        label_k_terbaik = [self.y_train[i] for i in k_terbaik]
        # voting yang paling banyak
        hasil_voting = Counter(label_k_terbaik).most_common(1)
        return hasil_voting[0][0]
    def jarak(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
        
    
@app.route("/")
def main():
    cur = mysql.connection.cursor()
    cur.execute("SELECT count(*) FROM ekstraksi_fitur")
    total_data = np.array(cur.fetchall())
    total_data = int(total_data)
    
    cur.execute("SELECT count(*) FROM uji")
    uji = np.array(cur.fetchall())
    uji = int(uji)
    
    cur.execute("SELECT count(*) FROM latih")
    latih = np.array(cur.fetchall())
    latih = int(latih)
    cur.close()
    return render_template("index.html", menu = 'dashboard', total_data = total_data, uji= uji, latih = latih)

@app.route("/ekstraksi_fitur")
def ekstraksi_fitur():
    cur = mysql.connection.cursor()
    cur.execute("SELECT*FROM ekstraksi_fitur")
    ekstraksi = cur.fetchall()
    cur.close()
    return render_template("ekstraksi_fitur.html", menu = 'ekstraksi_fitur', data = ekstraksi)

@app.route("/ekstraksi_aksi")
def ekstraksi_aksi():
    #Ekstraksi Data latih
    cursor= mysql.connection.cursor()
    cursor.execute("SELECT count(*) FROM ekstraksi_fitur")
    myresult = np.array(cursor.fetchall()) 
    jum_data = int(myresult[0][0])
    
    if jum_data == 0 :
    
        path = r"D:/Kampus/SMT8/belajar/ta/dataset_nila"
        image_fitur = []
        for file in os.listdir(path):
            imag=cv2.imread(os.path.abspath(path + "/" + file))
            resized_image = cv2.resize(imag, (417, 160))
            hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                

            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]

            totH = H.size
            totS = S.size
            totV = V.size

            totalH = H.sum()
            totalS = S.sum()
            totalV = V.sum()

            # Mean
            meanH = totalH/totH  
            meanS = totalS/totS  
            meanV = totalV/totV  

            # Stdev
            vrH = ((H - meanH)**2).sum()
            vrS = ((S - meanS)**2).sum()
            vrV = ((V - meanV)**2).sum()
            hasil_akhir_stdH = math.sqrt((vrH/totH))
            hasil_akhir_stdS = math.sqrt((vrS/totS))
            hasil_akhir_stdV = math.sqrt((vrV/totV))

            # Skewness
            meanijH = (H - meanH)
            meanijS = (S - meanS)
            meanijV = (V - meanV)

            skewnessH = (meanijH**3).sum()
            skewnessS = (meanijS**3).sum()
            skewnessV = (meanijV**3).sum()

            if skewnessH >= 0 :   
                hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
            elif skewnessH < 0 :
                hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
            if skewnessS >= 0 :
                hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
            elif skewnessS < 0 :
                hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
            if skewnessV >= 0 :
                hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
            elif skewnessV < 0 :
                hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                
            kelas = 0
            if file.startswith("segar") :
                kelas = 1
            else :
                kelas = 2
                
            fitur = [
                file,
                round(meanH,6),
                round(meanS,6),
                round(meanV,6),
                round(hasil_akhir_stdH,6),
                round(hasil_akhir_stdS,6),
                round(hasil_akhir_stdV,6),
                round(hasil_akhir_skewH,6),
                round(hasil_akhir_skewS,6),
                round(hasil_akhir_skewV,6),
                kelas       
            ]       
            
            cur = mysql.connection.cursor()
            sql = "INSERT INTO ekstraksi_fitur (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (fitur[0],fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8],fitur[9],fitur[10]) 
            cur.execute(sql, val)  
            mysql.connection.commit()
                           
    else:
        cursor= mysql.connection.cursor()
        cursor.execute("SELECT count(file_name) FROM ekstraksi_fitur where file_name LIKE 'segar%'")
        myresult_segar = np.array(cursor.fetchall()) 
        jum_segar = int(myresult_segar[0][0]) 
        
                 
        path = r"D:/Kampus/SMT8/belajar/ta/dataset_nila"
        tot_segar = str(jum_segar+1)
        try : 
            imag_segar= open(os.path.abspath(path + "/" + "segar"+ tot_segar + ".png"))
            
            if(imag_segar):
                # image_fitur = []
                imag=cv2.imread(os.path.abspath(path + "/" + "segar"+ str(jum_segar) + ".png"))
                nama = "segar"+str(tot_segar)+".png"
                resized_image = cv2.resize(imag, (417, 160))
                hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                    
                H = hsv[:, :, 0]
                S = hsv[:, :, 1]
                V = hsv[:, :, 2]

                totH = H.size
                totS = S.size
                totV = V.size

                totalH = H.sum()
                totalS = S.sum()
                totalV = V.sum()

                # Mean
                meanH = totalH/totH  
                meanS = totalS/totS  
                meanV = totalV/totV  

                # Stdev
                vrH = ((H - meanH)**2).sum()
                vrS = ((S - meanS)**2).sum()
                vrV = ((V - meanV)**2).sum()
                hasil_akhir_stdH = math.sqrt((vrH/totH))
                hasil_akhir_stdS = math.sqrt((vrS/totS))
                hasil_akhir_stdV = math.sqrt((vrV/totV))

                # Skewness
                meanijH = (H - meanH)
                meanijS = (S - meanS)
                meanijV = (V - meanV)

                skewnessH = (meanijH**3).sum()
                skewnessS = (meanijS**3).sum()
                skewnessV = (meanijV**3).sum()

                if skewnessH >= 0 :   
                    hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
                elif skewnessH < 0 :
                    hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
                if skewnessS >= 0 :
                    hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
                elif skewnessS < 0 :
                    hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
                if skewnessV >= 0 :
                    hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
                elif skewnessV < 0 :
                    hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                    
                kelas = 1
                
                    
                fitur = [
                    nama,
                    meanH,
                    meanS,
                    meanV,
                    hasil_akhir_stdH,
                    hasil_akhir_stdS,
                    hasil_akhir_stdV,
                    hasil_akhir_skewH,
                    hasil_akhir_skewS,
                    hasil_akhir_skewV,
                    kelas       
                ]   
                cur = mysql.connection.cursor()
                sql = "INSERT INTO ekstraksi_fitur (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (fitur[0],fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8],fitur[9],fitur[10]) 
                cur.execute(sql, val)  
                mysql.connection.commit()
                return redirect(url_for('ekstraksi_fitur'))
        except :                   
            cursor.execute("SELECT count(file_name) FROM ekstraksi_fitur where file_name LIKE 'non_segar%'")
            myresult_nonsegar = np.array(cursor.fetchall()) 
            jum_non = int(myresult_nonsegar[0][0])
            tot_nonsegar = str(jum_non+1)
            try:
                imag_non= open(os.path.abspath(path + "/" + "non_segar"+ tot_nonsegar + ".png"))
                if(imag_non):
                    # image_fitur = []
                    imag=cv2.imread(os.path.abspath(path + "/" + "non_segar"+ str(jum_non) + ".png"))
                    nama = "non_segar"+str(tot_nonsegar)+".png"
                    resized_image = cv2.resize(imag, (417, 160))
                    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                        
                    H = hsv[:, :, 0]
                    S = hsv[:, :, 1]
                    V = hsv[:, :, 2]

                    totH = H.size
                    totS = S.size
                    totV = V.size

                    totalH = H.sum()
                    totalS = S.sum()
                    totalV = V.sum()

                    # Mean
                    meanH = totalH/totH  
                    meanS = totalS/totS  
                    meanV = totalV/totV  

                    # Stdev
                    vrH = ((H - meanH)**2).sum()
                    vrS = ((S - meanS)**2).sum()
                    vrV = ((V - meanV)**2).sum()
                    hasil_akhir_stdH = math.sqrt((vrH/totH))
                    hasil_akhir_stdS = math.sqrt((vrS/totS))
                    hasil_akhir_stdV = math.sqrt((vrV/totV))

                    # Skewness
                    meanijH = (H - meanH)
                    meanijS = (S - meanS)
                    meanijV = (V - meanV)

                    skewnessH = (meanijH**3).sum()
                    skewnessS = (meanijS**3).sum()
                    skewnessV = (meanijV**3).sum()

                    if skewnessH >= 0 :   
                        hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
                    elif skewnessH < 0 :
                        hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
                    if skewnessS >= 0 :
                        hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
                    elif skewnessS < 0 :
                        hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
                    if skewnessV >= 0 :
                        hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
                    elif skewnessV < 0 :
                        hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                        
                    kelas = 2
                                    
                    fitur = [
                        nama,
                        meanH,
                        meanS,
                        meanV,
                        hasil_akhir_stdH,
                        hasil_akhir_stdS,
                        hasil_akhir_stdV,
                        hasil_akhir_skewH,
                        hasil_akhir_skewS,
                        hasil_akhir_skewV,
                        kelas       
                    ]    
                    
                    cur = mysql.connection.cursor()
                    sql = "INSERT INTO ekstraksi_fitur (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    val = (fitur[0],fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8],fitur[9],fitur[10]) 
                    cur.execute(sql, val)  
                    mysql.connection.commit()    
            except :
                return redirect(url_for('ekstraksi_fitur'))         
        
    return redirect(url_for('ekstraksi_fitur'))

@app.route("/pelatihan", methods = ["POST", "GET"])
def pelatihan():    
    cursor= mysql.connection.cursor()    
    cursor.execute("SELECT * FROM latih")
    q_hasil = cursor.fetchall()  
    cursor.close
    
    if request.method == 'POST' :
        nilai_k = request.form['nilai_k']
        if nilai_k == '':
            return render_template("pelatihan.html", menu = 'klasifikasi', submenu = 'pelatihan', data=q_hasil)

        X_train, X_test, y_train, y_test = spliting_data()
        model = KNN(k= nilai_k)
        model.train(X_train, y_train)
        x_pred = model.predict(X_train)
        akurasi = np.sum(x_pred == y_train)/len(X_train)
        cm = confusion_matrix(y_train, x_pred)
        benar = int(cm[0][0]) + int(cm[1][1])
        salah = int(cm[0][1]) + int(cm[1][0])
        
        return render_template("pelatihan.html", menu = 'datalatih', data=q_hasil, akurasi = round(akurasi,4), benar = benar, salah = salah)
    return render_template("pelatihan.html", menu = 'datalatih', data=q_hasil)
    

def data_latih():
    df = pd.read_sql_query("SELECT * FROM ekstraksi_fitur", cnx)
    x=df[['file_name','mean_h','mean_s','mean_v','stdev_h', 'stdev_s', 'stdev_v', 'skewness_h', 'skewness_s', 'skewness_v', 'class']].values
    Z_train,  W_train = train_test_split(x, test_size= 0.2, random_state= 42)
    latih = Z_train
    return latih

def data_uji():
    df = pd.read_sql_query("SELECT * FROM ekstraksi_fitur", cnx)
    x=df[['file_name','mean_h','mean_s','mean_v','stdev_h', 'stdev_s', 'stdev_v', 'skewness_h', 'skewness_s', 'skewness_v', 'class']].values
    Z_train,  W_train = train_test_split(x, test_size= 0.2, random_state= 42)
    uji = W_train
    return uji

def spliting_data():
    df = pd.read_sql_query("SELECT * FROM uji", cnx)
    df1 = pd.read_sql_query("SELECT * FROM latih", cnx)
    X=df[['mean_h','mean_s','mean_v','stdev_h', 'stdev_s', 'stdev_v', 'skewness_h', 'skewness_s', 'skewness_v']].values 
    Y=df['class'].values
    W=df1[['mean_h','mean_s','mean_v','stdev_h', 'stdev_s', 'stdev_v', 'skewness_h', 'skewness_s', 'skewness_v']].values 
    Z=df1['class'].values
    X_test = X
    y_test = Y
    X_train = W
    y_train = Z
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42)
    return X_train, X_test, y_train, y_test

@app.route("/tampildatauji")
def tampildatauji():
    df = pd.read_sql_query("SELECT * FROM uji", cnx)
    q_hasil = np.array(df)
    if request.method == 'POST' :
        nilai_k = request.form['nilai_k']
        if nilai_k == '':
            return render_template("tampildatauji.html", menu = 'datauji', data=q_hasil)      
        cursor= mysql.connection.cursor()    
        cursor.execute("SELECT * FROM uji")
        hasilnya= cursor.fetchall()
        cursor.close
        return render_template("tampildatauji.html", menu = 'datauji',  data=hasilnya)
    return render_template("tampildatauji.html", menu = 'datauji', data=q_hasil)

@app.route("/pengujian", methods = ["POST", "GET"])
def pengujian():
    df = pd.read_sql_query("SELECT * FROM uji", cnx)
    q_hasil = np.array(df)
    if request.method == 'POST' :
        nilai_k = request.form['nilai_k']
        if nilai_k == '':
            return render_template("pengujian.html", menu = 'klasifikasi', submenu = 'pengujian', data=q_hasil)

        X_train, X_test, y_train, y_test = spliting_data()
        model = KNN(k= nilai_k)
        model.train(X_train, y_train)
        hasil = model.predict(X_test)
        akurasi = np.sum(hasil == y_test)/len(X_test)
        cm = confusion_matrix(y_test, hasil)
        print(cm)
        benar = int(cm[0][0]) + int(cm[1][1])
        salah = int(cm[0][1]) + int(cm[1][0])
        
        cls_report = classification_report(y_test, hasil,output_dict=True)
        c_rpt = np.array([
                cls_report["1"]['precision'], cls_report["1"]['recall'], cls_report["1"]['f1-score'], cls_report["1"]['support'],
                cls_report["2"]['precision'], cls_report["2"]['recall'], cls_report["2"]['f1-score'], cls_report["2"]['support'],
                cls_report["macro avg"]['precision'], cls_report["macro avg"]['recall'], cls_report["macro avg"]['f1-score'], cls_report["macro avg"]['support'],
                cls_report["weighted avg"]['precision'], cls_report["weighted avg"]['recall'], cls_report["weighted avg"]['f1-score'], cls_report["weighted avg"]['support'] 
                ])
        akurasi = np.sum(hasil == y_test)/len(X_test)
        cursor= mysql.connection.cursor()    
        cursor.execute("SELECT * FROM uji")
        id_prt= cursor.fetchall() 
        idku = id_prt[0][0]
        for prediksi in hasil :
            cur = mysql.connection.cursor()
            sql = "UPDATE uji SET prediksi = %s WHERE id = %s"
            val = (prediksi,idku)
            cur.execute(sql, val)  
            mysql.connection.commit()
            idku+=1
        cursor= mysql.connection.cursor()    
        cursor.execute("SELECT * FROM uji")
        hasilnya= cursor.fetchall()
        cursor.close
        return render_template("pengujian.html", menu = 'klasifikasi', submenu = 'pengujian', data=hasilnya,
                            akurasi = round(akurasi,4), benar = benar, salah = salah,
                            precision1 = round(c_rpt[0],4), recall1 = round(c_rpt[1],4), f1_score1 = round(c_rpt[2],4), support1 = round(c_rpt[3],4),
                            precision2 = round(c_rpt[4],4), recall2 = round(c_rpt[5],4), f1_score2 = round(c_rpt[6],4), support2 = round(c_rpt[7],4),
                            precision3 = round(c_rpt[8],4), recall3 = round(c_rpt[9],4), f1_score3 = round(c_rpt[10],4),support3 = round(c_rpt[11],4),
                            precision4 = round(c_rpt[12],4), recall4 = round(c_rpt[13],4), f1_score4 = round(c_rpt[14],4), support4 = round(c_rpt[15],4))
    return render_template("pengujian.html", menu = 'klasifikasi', submenu = 'pengujian', data=q_hasil)

        
@app.route("/import_data_uji")
def import_data_uji():
    uji = data_uji()
    for citra in uji:        
        files = r"D:/Kampus/SMT8/belajar/ta/dataset_nila/"+ citra[0]
        gambar = cv2.imread(os.path.abspath(files))
        hasil = r"D:/Kampus/SMT8/belajar/ta/uploads/data_uji/"+ citra[0]
        cv2.imwrite(hasil, gambar)        
    cursor= mysql.connection.cursor()
    cursor.execute("SELECT count(*) FROM uji")
    hasil = np.array(cursor.fetchall()) 
    jum_data_uji= int(hasil[0][0])
    if jum_data_uji == 0:
        for d_uji in uji:
            cur = mysql.connection.cursor()
            sql = "INSERT INTO uji (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (d_uji[0], d_uji[1],d_uji[2],d_uji[3],d_uji[4],d_uji[5],d_uji[6],d_uji[7],d_uji[8],d_uji[9],d_uji[10]) 
            cur.execute(sql, val)  
            mysql.connection.commit()
    return redirect(url_for('tampildatauji'))

@app.route("/import_data_latih")
def import_data_latih():
    latih = data_latih()
    print(latih[0])
    for citra in latih:        
        files = r"D:/Kampus/SMT8/belajar/ta/dataset_nila/"+ citra[0]
        gambar = cv2.imread(os.path.abspath(files))
        hasil = r"D:/Kampus/SMT8/belajar/ta/uploads/data_latih/"+ citra[0]
        cv2.imwrite(hasil, gambar)        
    cursor= mysql.connection.cursor()
    cursor.execute("SELECT count(*) FROM latih")
    hasil = np.array(cursor.fetchall()) 
    jum_data_latih= int(hasil[0][0])
    if jum_data_latih == 0:
        for d_latih in latih:
            cur = mysql.connection.cursor()
            sql = "INSERT INTO latih (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (d_latih[0], d_latih[1],d_latih[2],d_latih[3],d_latih[4],d_latih[5],d_latih[6],d_latih[7],d_latih[8],d_latih[9],d_latih[10]) 
            cur.execute(sql, val)  
            mysql.connection.commit()
    return redirect(url_for('pelatihan'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route("/tambah_data_uji", methods = ["POST", "GET"])
def tambah_data_uji():
    if request.method == 'POST':  
        file = request.files['file']
        kesegaran = request.form['kesegaran']
              
        if 'file' not in request.files:
            return redirect(url_for('tampildatauji'))
        
        if file.filename == '':
            return redirect(url_for('tampildatauji'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            direc = r"D:/Kampus/SMT8/belajar/ta/uploads/data_uji/"
            file.save(os.path.join(direc , filename))
            image = cv2.imread(direc+filename)
            if os.path.isfile(direc+filename): 
                resized_image = cv2.resize(image, (417, 160))
                hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                H = hsv[:, :, 0]
                S = hsv[:, :, 1]
                V = hsv[:, :, 2]

                totH = H.size
                totS = S.size
                totV = V.size

                totalH = H.sum()
                totalS = S.sum()
                totalV = V.sum()
                        
                # Mean
                meanH = totalH/totH  
                meanS = totalS/totS  
                meanV = totalV/totV  

                # Stdev
                vrH = ((H - meanH)**2).sum()
                vrS = ((S - meanS)**2).sum()
                vrV = ((V - meanV)**2).sum()
                hasil_akhir_stdH = math.sqrt((vrH/totH))
                hasil_akhir_stdS = math.sqrt((vrS/totS))
                hasil_akhir_stdV = math.sqrt((vrV/totV))
                        
                # Skewness
                meanijH = (H - meanH)
                meanijS = (S - meanS)
                meanijV = (V - meanV)

                skewnessH = (meanijH**3).sum()
                skewnessS = (meanijS**3).sum()
                skewnessV = (meanijV**3).sum()
                        
                if skewnessH >= 0 :   
                    hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
                elif skewnessH < 0 :
                    hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
                if skewnessS >= 0 :
                    hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
                elif skewnessS < 0 :
                    hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
                if skewnessV >= 0 :
                    hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
                elif skewnessV < 0 :
                    hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                
                          
                fitur = [
                    round(meanH,6),
                    round(meanS,6),
                    round(meanV,6),
                    round(hasil_akhir_stdH,6),
                    round(hasil_akhir_stdS,6),
                    round(hasil_akhir_stdV,6),
                    round(hasil_akhir_skewH,6),
                    round(hasil_akhir_skewS,6),
                    round(hasil_akhir_skewV,6),   
                ]
                
                cur = mysql.connection.cursor()
                sql = "INSERT INTO uji (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (filename, fitur[0], fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8], kesegaran) 
                cur.execute(sql, val)  
                mysql.connection.commit()    
                return redirect(url_for('tampildatauji'))
            return redirect(url_for('tampildatauji'))  
        return redirect(url_for('tampildatauji'))          
    return redirect(url_for('tampildatauji'))

@app.route("/tambah_data_latih", methods = ["POST", "GET"])
def tambah_data_latih():
    if request.method == 'POST':  
        file = request.files['file']
        kesegaran = request.form['kesegaran']
              
        if 'file' not in request.files:
            return redirect(url_for('pelatihan'))
        
        if file.filename == '':
            return redirect(url_for('pelatihan'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            direc = r"D:/Kampus/SMT8/belajar/ta/uploads/data_latih/"
            file.save(os.path.join(direc , filename))
            image = cv2.imread(direc+filename)
            if os.path.isfile(direc+filename): 
                resized_image = cv2.resize(image, (417, 160))
                hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                H = hsv[:, :, 0]
                S = hsv[:, :, 1]
                V = hsv[:, :, 2]

                totH = H.size
                totS = S.size
                totV = V.size

                totalH = H.sum()
                totalS = S.sum()
                totalV = V.sum()
                        
                # Mean
                meanH = totalH/totH  
                meanS = totalS/totS  
                meanV = totalV/totV  

                # Stdev
                vrH = ((H - meanH)**2).sum()
                vrS = ((S - meanS)**2).sum()
                vrV = ((V - meanV)**2).sum()
                hasil_akhir_stdH = math.sqrt((vrH/totH))
                hasil_akhir_stdS = math.sqrt((vrS/totS))
                hasil_akhir_stdV = math.sqrt((vrV/totV))
                        
                # Skewness
                meanijH = (H - meanH)
                meanijS = (S - meanS)
                meanijV = (V - meanV)

                skewnessH = (meanijH**3).sum()
                skewnessS = (meanijS**3).sum()
                skewnessV = (meanijV**3).sum()
                        
                if skewnessH >= 0 :   
                    hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
                elif skewnessH < 0 :
                    hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
                if skewnessS >= 0 :
                    hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
                elif skewnessS < 0 :
                    hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
                if skewnessV >= 0 :
                    hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
                elif skewnessV < 0 :
                    hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                
                          
                fitur = [
                    round(meanH,6),
                    round(meanS,6),
                    round(meanV,6),
                    round(hasil_akhir_stdH,6),
                    round(hasil_akhir_stdS,6),
                    round(hasil_akhir_stdV,6),
                    round(hasil_akhir_skewH,6),
                    round(hasil_akhir_skewS,6),
                    round(hasil_akhir_skewV,6),   
                ]
                
                cur = mysql.connection.cursor()
                sql = "INSERT INTO latih (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (filename, fitur[0], fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8], kesegaran) 
                cur.execute(sql, val)  
                mysql.connection.commit()    
                return redirect(url_for('pelatihan'))
            return redirect(url_for('pelatihan'))  
        return redirect(url_for('pelatihan'))          
    return redirect(url_for('pelatihan'))

@app.route("/prediksi", methods = ["POST", "GET"])
def prediksi():
    df = pd.read_sql_query("SELECT * FROM latih", cnx)
    df1 = pd.read_sql_query("SELECT * FROM prediksi", cnx)
    q_hasil = np.array(df)
    prediksi = np.array(df1)
    if request.method == 'POST':
        file = request.files['file']
        nilai_k = request.form['nilai_k']
        X_train, X_test, y_train, y_test = spliting_data()
        model = KNN(k= nilai_k)
        model.train(X_train, y_train)
        if 'file' not in request.files:
            return redirect(url_for('prediksi'))
        
        if file.filename == '':
            return redirect(url_for('prediksi'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            direc = r"D:/Kampus/SMT8/belajar/ta/uploads/data_prediksi/"
            file.save(os.path.join(direc , filename))
            image = cv2.imread(direc+filename)
            if os.path.isfile(direc+filename): 
                resized_image = cv2.resize(image, (417, 160))
                hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
                H = hsv[:, :, 0]
                S = hsv[:, :, 1]
                V = hsv[:, :, 2]

                totH = H.size
                totS = S.size
                totV = V.size

                totalH = H.sum()
                totalS = S.sum()
                totalV = V.sum()
                        
                # Mean
                meanH = totalH/totH  
                meanS = totalS/totS  
                meanV = totalV/totV  

                # Stdev
                vrH = ((H - meanH)**2).sum()
                vrS = ((S - meanS)**2).sum()
                vrV = ((V - meanV)**2).sum()
                hasil_akhir_stdH = math.sqrt((vrH/totH))
                hasil_akhir_stdS = math.sqrt((vrS/totS))
                hasil_akhir_stdV = math.sqrt((vrV/totV))
                        
                # Skewness
                meanijH = (H - meanH)
                meanijS = (S - meanS)
                meanijV = (V - meanV)

                skewnessH = (meanijH**3).sum()
                skewnessS = (meanijS**3).sum()
                skewnessV = (meanijV**3).sum()
                        
                if skewnessH >= 0 :   
                    hasil_akhir_skewH = math.pow(skewnessH/totH, float(1)/3)
                elif skewnessH < 0 :
                    hasil_akhir_skewH = -np.float_power(abs(skewnessH/totH), float(1)/3)
                if skewnessS >= 0 :
                    hasil_akhir_skewS = math.pow(skewnessS/totS, float(1)/3)
                elif skewnessS < 0 :
                    hasil_akhir_skewS = -np.float_power(abs(skewnessS/totS), float(1)/3)
                if skewnessV >= 0 :
                    hasil_akhir_skewV = math.pow(skewnessV/totV, float(1)/3)
                elif skewnessV < 0 :
                    hasil_akhir_skewV =  -np.float_power(abs(skewnessV/totV), float(1)/3)
                
                          
                fitur = [
                    round(meanH,4),
                    round(meanS,4),
                    round(meanV,4),
                    round(hasil_akhir_stdH,4),
                    round(hasil_akhir_stdS,4),
                    round(hasil_akhir_stdV,4),
                    round(hasil_akhir_skewH,4),
                    round(hasil_akhir_skewS,4),
                    round(hasil_akhir_skewV,4),   
                ]
                
                citra = [[
                        fitur[0],
                        fitur[1],
                        fitur[2],
                        fitur[3],
                        fitur[4],
                        fitur[5],
                        fitur[6],
                        fitur[7],
                        fitur[8]
                        ]]
                y_pred = model.predict(citra)
                y_pred = int(y_pred[0])
                
                cur = mysql.connection.cursor()
                sql = "INSERT INTO prediksi (file_name, mean_h, mean_s, mean_v, stdev_h, stdev_s, stdev_v, skewness_h, skewness_s, skewness_v, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (filename, fitur[0], fitur[1],fitur[2],fitur[3],fitur[4],fitur[5],fitur[6],fitur[7],fitur[8], y_pred) 
                cur.execute(sql, val)  
                mysql.connection.commit()
                
                cursor= mysql.connection.cursor()    
                cursor.execute("SELECT * FROM latih")
                id_prt= cursor.fetchall() 
                idku = id_prt[0][0]
                jarak = np.array(knn_jarak).flatten()
                for jr in jarak :
                    cur = mysql.connection.cursor()
                    sql = "UPDATE latih SET jarak = %s WHERE id = %s"
                    val = (round(jr,4),idku)
                    cur.execute(sql, val)  
                    mysql.connection.commit()
                    idku+=1
                cursor= mysql.connection.cursor()    
                cursor.execute("SELECT * FROM latih")
                hasilnya= cursor.fetchall()
                cursor.close                               
                cursor= mysql.connection.cursor()    
                cursor.execute("SELECT * FROM prediksi")
                prediksi= cursor.fetchall()
                cursor.close                               
        return render_template("prediksi.html", menu = 'klasifikasi', submenu = 'prediksi', y_pred = y_pred, data = hasilnya, prediksi = prediksi)    
    return render_template("prediksi.html", menu = 'klasifikasi', submenu = 'prediksi', data = q_hasil, prediksi = prediksi)

@app.route("/c_report", methods = ["POST", "GET"])
def c_report():
    if request.method == 'POST':
        nilai_k = request.form['nilai_k']
        print(nilai_k)
        X_train, X_test, y_train, y_test = spliting_data()
    
        model = KNN(k= nilai_k)
        model.train(X_train, y_train)
        hasil = model.predict(X_test)
        cls_report = classification_report(y_test, hasil,output_dict=True)
        c_rpt = np.array([
                cls_report["1"]['precision'], cls_report["1"]['recall'], cls_report["1"]['f1-score'], cls_report["1"]['support'],
                cls_report["2"]['precision'], cls_report["2"]['recall'], cls_report["2"]['f1-score'], cls_report["2"]['support'],
                cls_report["macro avg"]['precision'], cls_report["macro avg"]['recall'], cls_report["macro avg"]['f1-score'], cls_report["macro avg"]['support'],
                cls_report["weighted avg"]['precision'], cls_report["weighted avg"]['recall'], cls_report["weighted avg"]['f1-score'], cls_report["weighted avg"]['support'] 
                ])
        akurasi = np.sum(hasil == y_test)/len(X_test)
        return render_template(
            "c_report.html", menu = 'evaluasi', submenu = 'laporan_klasifikasi', 
            precision1 = round(c_rpt[0],4), recall1 = round(c_rpt[1],4), f1_score1 = round(c_rpt[2],4), support1 = round(c_rpt[3],4),
            precision2 = round(c_rpt[4],4), recall2 = round(c_rpt[5],4), f1_score2 = round(c_rpt[6],4), support2 = round(c_rpt[7],4),
            precision3 = round(c_rpt[8],4), recall3 = round(c_rpt[9],4), f1_score3 = round(c_rpt[10],4),support3 = round(c_rpt[11],4),
            precision4 = round(c_rpt[12],4), recall4 = round(c_rpt[13],4), f1_score4 = round(c_rpt[14],4), support4 = round(c_rpt[15],4), akurasi = round(akurasi,4))
    return render_template("c_report.html", menu = 'evaluasi', submenu = 'laporan_klasifikasi')

@app.route("/report")
def report():
    
            
    return render_template("report.html", menu = 'evaluasi', submenu = 'laporan')
        


if __name__ == "__main__":
    app.run(debug=True)