#!/usr/bin/python
# -*- coding: UTF-8 -*-
from flask import Flask, request, jsonify
import urllib
import VehicleLicenceRecognize as vlr
import json
from flask import render_template


app = Flask(__name__)

#对于imgurl需要进行encode编码，否则会出错！
@app.route('/vehicle/licence/recognize', methods=['post','get'])
def recignize():
    p = request.args.get('imgurl')

    if p == None:
    	return jsonify({'t': p})

    imgurl = urllib.unquote(p)
    ret = vlr.licenceRecognize(imgurl)  
    return ret

#imgurl需要进行encode编码，否则会出错！
@app.route('/vehicle/licence/recognize1', methods=['post','get'])
def recignize1():
    p = request.args.get('imgurl')

    if p == None:
        return jsonify({'t': p})

    imgurl = urllib.unquote(p)
    ret = vlr.licenceRecognize(imgurl)
    print ret
    return render_template("licence.html",
        title = 'myvehicle',
        licence = json.loads(ret))

@app.errorhandler(404)
def not_found(error):
    return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
