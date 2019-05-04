from flask import Flask, request, jsonify
from model_predict import model
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

class ASAGGrader(Resource):
    def get(self):
        q,ra,sa,mf = "","","",""
        if 'quest' in request.args:
            q = request.args.get('quest', None)
        if 'ref_ans' in request.args:
            ra = request.args.get('ref_ans', None)
        if 'stu_ans' in request.args:
            sa = request.args.get('stu_ans', None)
        if 'model' in request.args:
            mf = request.args.get('model', None)
        grade = model(q,ra,sa,mf)
        return {'graded':grade}

api.add_resource(ASAGGrader, '/asag_api')

if __name__ == '__main__':
    app.run(debug=False)
