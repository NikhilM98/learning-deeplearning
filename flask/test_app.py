from flask import Flask
app = Flask(__name__)

@app.route('/test')
def running():
    return 'Flask is up and running!'
