from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Criar a instância do Flask
app = Flask(__name__)

# Rota principal
@app.route('/')
def home():
    return "Olá, Flask está rodando!"

# Rota para a funcionalidade de vetorizar o PDF
@app.route('/vetorizar', methods=['POST'])
def vetorizar_pdf():
    try:
        # Receber dados do JSON
        data = request.get_json()
        file_url = data.get('file_url')  # URL do PDF
        condominio_id = data.get('condominio_id')  # ID do condomínio

        if not file_url or not condominio_id:
            return jsonify({"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}), 400

        # Aqui você adicionaria o código para processar o PDF e gerar os embeddings
        # Isso pode ser feito chamando a função que você já tem para vetorizar o PDF

        return jsonify({"message": "Vetorização completada com sucesso!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Iniciar o servidor
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
