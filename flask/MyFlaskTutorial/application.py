from flask import Flask, render_template, request

app = Flask(__name__)
app.vars = {}

@app.route('/index', methods=['GET','POST'])
def index():
	nquestions = 5
	if request.method == 'GET':
		return render_template('userinfo.html', num=nquestions)
	else:
		#request was a POST
		app.vars['name'] = request.form['name']
		app.vars['age'] = request.form['age']

		f = open('%s_%s.txt'%(app.vars['name'],app.vars['age']),'w')
		f.write('Name: %s\n'%(app.vars['name']))
		f.write('Age: %s\n\n'%(app.vars['age']))
		f.close()

		return render_template('layout.html', num=1,
			question='How many eyes do you have?', ans1='1', ans2='2', ans3='3')

if __name__ == '__main__':
	app.run(debug=True)