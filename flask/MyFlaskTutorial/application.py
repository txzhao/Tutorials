from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/index', methods=['GET','POST'])
def index():
	nquestions = 5
	if request.method == 'GET':
		return render_template('userinfo.html', num=nquestions)
	else:
		return 'request.method was not a GET!'

if __name__ == '__main__':
	app.run(debug=True)