from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from model_predict import model
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)

class ASAGGrader(Resource):
    def post(self):
        q,ra,sa,mf = "","","",""
        content = request.get_json()
        try:
            q = content['quest']
            ra = content['ref_ans']
            sa = content['stu_ans']
            mf = content['model']
        except:
            pass
        grade = model(q,ra,sa,mf)
        return {'quest':q,'ref_ans':ra,'stu_ans':sa,'mtype':mf,'grade':grade}

api.add_resource(ASAGGrader, '/')

if __name__ == '__main__':
    app.run()
