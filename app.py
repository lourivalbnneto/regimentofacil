from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask está funcionando!"

@app.route('/vetorizar', methods=['POST'])
def vetorizar_pdf():
    try:
        data = request.get_json()
        file_url = data.get('file_url')
        condominio_id = data.get('condominio_id')

        if not file_url or not condominio_id:
            return jsonify({"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}), 400

        # Lógica de vetorização

        return jsonify({"message": "Vetorização completada com sucesso!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
