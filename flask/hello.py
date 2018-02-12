from flask import Flask
app = Flask(__name__)

@app.route('/hello_page')
def hello_world():
	# note that the function name and the route argument
    # do not need to be the same.
	return 'Hello World!'

if __name__ == '__main__':
	app.run(debug=True)