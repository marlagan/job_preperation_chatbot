from flask import Flask, request, jsonify, render_template
from interaction_with_user import answer
app = Flask(__name__,  template_folder='frontend', static_folder='frontend')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    message = data.get('message')
    an = answer(message)
    return jsonify(an)


if __name__ == '__main__':
    app.run(port=13678, debug=True)


