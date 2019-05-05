from flask_restful import Resource, Api, reqparse
from flask import Flask, request, jsonify
from model_predict import model
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)
parser = reqparse.RequestParser()

class ASAGGrader(Resource):
    def post(self):
        parser.add_argument('quest',type=str)
        parser.add_argument('ref_ans',type=str)
        parser.add_argument('stu_ans',type=str)
        parser.add_argument('model',type=str)
        q,ra,sa,mf = "","","",""
        args = parser.parse_args()
        q = args['quest']
        ra = args['ref_ans']
        sa = args['stu_ans']
        mf = args['model']
        grade = model(q,ra,sa,mf)
        return {'quest':q,'ref_ans':ra,'stu_ans':sa,'model':mf,'grade':grade}

api.add_resource(ASAGGrader, '/')

if __name__ == '__main__':
    app.run()
