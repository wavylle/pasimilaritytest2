# A very simple Flask Hello World app for you to get started with...

from flask import Flask, jsonify
from flask import request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route("/pachecksimilarity", methods=['POST'])
def pachecksimilarity():
    try:
        data = request.get_json()

        s1 = data.get("s1")
        s1_emb = model.encode(s1)

        queryData = data.get("data")
        queryData = queryData[:200]

        score_dict = queryData
        
        for e, x in enumerate(queryData):
            score_dict[e]["similarity_score"] = cosine_similarity(s1_emb.reshape(1, -1),model.encode(x["query"]).reshape(1, -1))[0][0]

        max_similarity_score = 0

        if len(queryData) > 0:
            max_similarity_score = max(item['similarity_score'] for item in score_dict)

        index_with_max_score = max(range(len(score_dict)), key=lambda i: score_dict[i]['similarity_score'])
        response_with_max_score = score_dict[index_with_max_score]['response']

        json_message = {"status": "success", "score": str(max_similarity_score), "response": str(response_with_max_score)}

        return jsonify(json_message)
    except Exception as e:
        json_message = {"status": "failure", "error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)
