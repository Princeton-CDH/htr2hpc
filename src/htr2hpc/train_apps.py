import pathlib
from zipfile import ZipFile

import parsl
from parsl.app.app import python_app, bash_app
from parsl.data_provider.files import File

from htr2hpc.api_client import eScriptoriumAPIClient


@python_app
def prep_training_data(es_base_url, es_api_token, document_id, transcription_id):
    from htr2hpc.api_client import eScriptoriumAPIClient

    api = eScriptoriumAPIClient(es_base_url, api_token=es_api_token)
    # get current user info, since id is needed to determine export filename
    user = api.current_user()
    # get document details so we don't have to specify both document id and name
    document = api.document(document_id)
    export_zipfile = api.download_document_export(
        user.pk, document_id, document.name, transcription_id
    )
    # run in current working directory for now
    data_dir = pathlib.Path("training_data")
    data_dir.mkdir(exist_ok=True)

    # extract everything in the zipfiile to the training data dir
    with ZipFile(export_zipfile) as zip_exp:
        zip_exp.extractall(path=data_dir)

    # return training directory as a parsl file
    return File(data_dir)


@bash_app
def segtrain(inputs=[], outputs=[]):
    # first input should be directory for input data
    input_data_dir = inputs[0]
    # TODO: real args here
    return f"ketos segtrain -f alto {input_data_dir}/0*.xml"
    # TODO: return model as output file


# use api.update_model with model id and pathlib.Path to model file
# to update existing model record with new file
