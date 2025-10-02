# import envparse
from defender.apps import create_app

# CUSTOMIZE: import model to be used
from defender.models.whitebox_mlp_model import WhiteboxMLPEmberModel

if __name__ == "__main__":
    # retrive config values from environment variables
    # model_gz_path = envparse.env("DF_MODEL_GZ_PATH", cast=str, default="models/ember_model.txt.gz")

    # CUSTOMIZE: app and model instance

    model = WhiteboxMLPEmberModel("models/whitebox_mlp.pt")


    app = create_app(model)

    import sys
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 8080

    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()

    # curl -XPOST --data-binary @somePEfile http://127.0.0.1:8080/ -H "Content-Type: application/octet-stream"
