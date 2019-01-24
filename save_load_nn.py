import keras


def save_nn(nn_model, name="model"):
    # serialize model to JSON
    model_json = nn_model.to_json()
    with open("Saved_models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights("Saved_models/" + name + ".h5")
    print("Saved model " + name + " to disk")


def load_nn(name="model", repository=keras):
    # load json and create model
    json_file = open("Saved_models/" + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = repository.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Saved_models/" + name + ".h5")
    print("Loaded model " + name + " from disk")
    return loaded_model
