from flask import Flask, request, render_template, session
import matplotlib.pyplot as plt
from datapreprocessing import DataPreProcessing
from spin import SpInML
from spinmlviz import SpInMLViz
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key="thisiskey"

@app.route('/')
def index():
    return render_template('index.html') #,cp=cpdata,lu=ludata)

@app.route('/home')
def home():
    obj = SpInML()
    inj_df, gmw_df, met_df, mgd_df = obj.getDatasets()
    return render_template('home.html',id=inj_df,gd=gmw_df,md=met_df,mg=mgd_df)

@app.route('/datasetview')
def datasetview():
    obj = SpInML()
    inj_df, gmw_df, met_df, mgd_df = obj.getConvertedData()
    inj  = inj_df.head(8).to_html()
    gmw = gmw_df.head(8).to_html()
    met = met_df.head(8).to_html()
    mgd = mgd_df.head(8).to_html()
    #
    # file = open("templates/injuries.html", "w")
    # file.write(inj)
    # file.close()
    # file = open("templates/workoverload.html", "w")
    # file.write(gmw)
    # file.close()
    #
    # file = open("templates/metrics.html", "w")
    # file.write(met)
    # file.close()
    # file = open("templates/merged.html", "w")
    # file.write(mgd)
    # file.close()
    return render_template('datasetview.html') #,cp=cph,lu=luh)

@app.route('/datasetdetails')
def datasetdetails():
    return render_template('datasetdetails.html') #,cp=cpdata,lu=lpdata,ch=cheader,lh=lpheader)

@app.route('/statdetails')
def statdetails():
    obj = SpInML()
    inj_df, gmw_df, met_df, mgd_df = obj.getConvertedData()
    inj = inj_df.describe().to_html()
    gmw = gmw_df.describe().to_html()
    met = met_df.describe().to_html()
    mgd = mgd_df.describe().to_html()
    # ces = cp.describe()
    # les = lu.describe()
    #
    # cph = ces.to_html()
    # luh = les.to_html()
    #
    # file = open("templates/injuriesD.html", "w")
    # file.write(inj)
    # file.close()
    # file = open("templates/workoverloadD.html", "w")
    # file.write(gmw)
    # file.close()
    #
    # file = open("templates/metricsD.html", "w")
    # file.write(met)
    # file.close()
    # file = open("templates/mergedD.html", "w")
    # file.write(mgd)
    # file.close()
    return render_template('statdetails.html')#,cp=ces,lu=lu)

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/clustering')
def clustering():
    dp = DataPreProcessing()
    dp.dataConversion()
    m, a = dp.classificationModels()
    return render_template('clustering.html',m=m,a=a)

@app.route('/classification')
def classification():
    return render_template('classification.html')#,ts=ts,ds=datas)

@app.route('/frequency')
def frequency():
    return render_template('frequency.html')#,df=df,t=t)

@app.route('/distribution')
def distribution():
    return render_template('distribution.html')

@app.route('/dataviz')
def dataviz():
    return render_template('dataviz.html')

@app.route('/terminology')
def terminology():
    return render_template('terminology.html')

@app.route('/finaldata')
def finaldata():
    return render_template('finaldata.html')

@app.route('/preprocessing')
def preprocessing():
    dp = DataPreProcessing()
    d1,d2,d3,d4 = dp.dataConversion()
    x1 = d1.head(8).to_html()
    x2 = d2.head(8).to_html()
    x3 = d3.head(6).to_html()
    x4 = d4.head(15).to_html()
    #
    file = open("templates/pp1.html", "w")
    file.write(x1)
    file.close()
    file = open("templates/pp2.html", "w")
    file.write(x2)
    file.close()

    file = open("templates/pp3.html", "w")
    file.write(x3)
    file.close()
    file = open("templates/pp4.html", "w")
    file.write(x4)
    file.close()
    return render_template('preprocessing.html')

@app.route('/algorithms')
def algorithms():
    return render_template('algorithms.html')

@app.route('/compana')
def compana():
    # obj = TwitterData()
    # cpdata,ludata = obj.getDataSet()
    # cs = Classification()
    # ts = []
    # datas = []
    # data, trset, tsset, conf = cs.decesionTree()
    # ts.append(tsset)
    # datas.append(data)
    # data, trset, tsset, conf = cs.linearRegression()
    # ts.append(tsset)
    # datas.append(data)
    # data, trset, tsset, conf = cs.svm()
    # ts.append(tsset)
    # datas.append(data)
    # data, trset, tsset, conf = cs.naiveBayes()
    # ts.append(tsset)
    # datas.append(data)
    # data, trset, tsset, conf = cs.newClassifier()
    # ts.append(tsset*1.2)
    # datas.append(data)
    # # d = ['DT','LR','SVM','NB','NC']
    # # plt.plot(d,ts)
    # # plt.plot(d, ts,'r^')
    # # plt.grid()
    # # plt.xlabel('Classification Method')
    # # plt.ylabel('Training Set Accuracy')
    # # plt.title('Comparision of Accuracy on Classfication Methods')
    # # #plt.show()
    # # # import os
    # # # os.remove('static/images/finalcomp.png')
    # # plt.savefig('static/images/finalcomp.png')
    # # ts.clear()
    return render_template('compana.html')

if __name__ == '__main__':
    app.run(debug=True)
