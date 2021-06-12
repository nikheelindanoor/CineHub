from flask import Flask, render_template, request, jsonify
from api import listofmovies
import os
import json
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    if request.method == 'POST': 
        movie_response = request.get_json()
        try:
            movie_input = movie_response["movie_input"]
            title = movie_input["movie"]
            data = listofmovies(title)
            strdata = json.dumps(data)
        except:
            strdata = " "

        return jsonify({"data": strdata})
    else:
        return render_template("index.html")

PORT = int(os.environ.get('PORT', 5000))
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=PORT)