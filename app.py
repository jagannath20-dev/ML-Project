from flask import Flask, render_template,request
import os
import pandas as pd
import numpy as np
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__)

@app.route('/train',methods=['GET']) #route to train the pipeline
def hello_world():
    return "<p>Hello, World!<p>"                                                          

if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)
