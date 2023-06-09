import os
import re
from collections import namedtuple

import requests
from ipywidgets import Output
from traitlets import HasTraits, TraitType

RegisteredWidget = namedtuple("RegisteredWidget", ["name", "widget"])


class AppState(HasTraits):
    widgets: dict

    @property
    def traits(self):
        return list(self.__dict__.keys())

    def to_dict(self):
        return self.__dict__

    def register_stateful_widget(
        self, widget: any, trait_name: str, traitlet: TraitType
    ):
        if not hasattr(self, "widgets"):
            self.widgets = {}
        self.add_traits(**{trait_name: traitlet})
        self.widgets[trait_name] = widget
        self.widgets[trait_name].observe(
            lambda _: self.update_from_ui(), names=["value"]
        )

    def register_stateful_property(self, trait_name: str, traitlet: TraitType):
        self.add_traits(**{trait_name: traitlet})

    def register_widget_observer(self, observes: str, observer):
        self.widgets[observes].observe(observer, names=["value"])

    def update_from_ui(self):
        for name in self.widgets.keys():
            setattr(self, name, self.widgets[name].value)

    def restore_ui(self):
        for name in self.widgets.keys():
            self.widgets[name].value = self[name]

    def observe_trait(self, trait_name: str, fn):
        def observer(state):
            if state["name"] == trait_name:
                fn(state)

        return self.observe(observer)

    @property
    def outlet(self):
        out = Output()

        def print_state(s):
            with out:
                print(s)

        self.observe(print_state)
        with out:
            print("AppState Outlet:")
        return out


def validate_upload_specifications(list_of_things_to_upload):
    expected_keys = ["local_filepath", "name", "description", "content_type"]
    for upload in list_of_things_to_upload:
        if not all(key in upload for key in expected_keys):
            raise ValueError(
                f"missing key in {upload} - expected all of {expected_keys}"
            )
        if not os.path.exists(upload["local_filepath"]):
            raise ValueError(f"missing local file: {upload['local_filepath']}")


def upload_a_file(filepath: str, name: str, content_type: str):
    # 1. get a signed url to upload to
    url = f'{os.environ["RECHARGE_API_ENDPOINT"]}/upload'
    filesize = os.stat(filepath).st_size
    r = requests.post(
        url,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['RECHARGE_API_TOKEN']}",
        },
        json={
            "filename": re.sub("[_/.]", "-", name),
            "content_type": content_type,
            "size": filesize,
        },
    )

    if r.status_code >= 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")

    upload_info = r.json()

    # 2. Upload the file.
    r = requests.put(
        upload_info["url"],
        headers={
            "content-type": content_type,
            "Content-Range": "bytes 0-" + str(filesize - 1) + "/" + str(filesize),
        },
        data=open(filepath, "rb"),
    )

    if r.status_code > 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")

    return {"id": upload_info["id"], "size": filesize}


def add_exports_to_simulation(simulationId, list_of_exports):
    url = f'{os.environ["RECHARGE_API_ENDPOINT"]}/simulations/{simulationId}/exports'
    r = requests.patch(
        url,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['RECHARGE_API_TOKEN']}",
        },
        json={"items": list_of_exports},
    )

    if r.status_code >= 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")


def upload_and_export_all(list_of_things_to_upload):
    print(f"Uploading {len(list_of_things_to_upload)} outputs")
    errors = []
    exports_to_add = []
    for upload in list_of_things_to_upload:
        print(f"Uploading {upload['name']}...", end="", flush=True)
        try:
            upload_info = upload_a_file(
                upload["local_filepath"], upload["name"], upload["content_type"]
            )
            exports_to_add.append({**upload, **upload_info})
            print("DONE")
        except Exception as err:
            print("ERROR!")
            errors.append(err)

    print("File uploads completed!")

    if len(errors) > 0:
        print(f"{len(errors)} errors encountered")
        for n, err in enumerate(errors):
            print(f"{n}: {err}")

    print("Upload success!")
    print("Updating simulation...")
    add_exports_to_simulation(os.environ["SIMULATION_ID"], exports_to_add)
    print("Update completed!")

def fetch_user_dataset(dataset_id):
    url = f'{os.environ["RECHARGE_API_ENDPOINT"]}/datasets/{dataset_id}'
    
    r = requests.get(
        url,
        headers={
            "authorization": f"Bearer {os.environ['RECHARGE_API_TOKEN']}",            
        }
    )
    
    if r.status_code >= 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")
    
    return r.json()

    
    
def download_user_dataset_file(filename, file_info, output_folder):
    r = requests.get(file_info["url"])

    if r.status_code >= 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")

    if file_info['content_type'] == 'application/geo+json':
        ext = 'geojson'
    elif file_info['content_type'] == 'text/csv':
        ext = 'csv'
    else:
        ext = 'txt'

    filepath = os.path.join(output_folder, f"{filename}.{ext}")
    print(f"Writing {file_info['size']} bytes to {filename}.{ext}...")
    with open(filepath, "w") as f:
        f.write(r.text)
    
def download_user_sediment_type_data(dataset, output_folder):
    if dataset["files"]["sediment_type"] is None:
        raise ValueError(f"Error expected dataset to have sedmiment type data")
    
    download_user_dataset_file("sediment_type", dataset["files"]["sediment_type"], output_folder)
    

def download_all_user_dataset_files(dataset, output_folder):    
    [download_user_dataset_file(k,v,output_folder) for k,v in dataset["files"].items()]


def create_dataset(list_of_uploads, dataset_name, em_filename, wells_filename, survey_year=None, survey_season=None, survey_kind=None):
    url = f'{os.environ["RECHARGE_API_ENDPOINT"]}/datasets';
    
    data = dict(
        name=dataset_name,
        files={item['key']: item['id'] for item in list_of_uploads}
    )
    
    if em_filename is not None:
        data["original_em_filename"] = em_filename
        if survey_year is None or survey_season is None or survey_kind is None:
            raise ValueError(f"Error expected survey_year, survey_season, and survey_kind to be set")
        data["survey_year"] = survey_year
        data["survey_season"] = survey_season
        data["survey_kind"] = survey_kind
    
    if wells_filename is not None:
        data["original_sediment_type_filename"] = wells_filename        

    r = requests.post(
        url,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['RECHARGE_API_TOKEN']}",
        },
        json=data,
    )
    
    print(f"Posted {len(list_of_uploads)} files")

    if r.status_code >= 299:
        raise ValueError(f"Error {r.status_code} - {r.text}")
        
    dataset = r.json()

    print(f'Dataset Created with ID: {dataset["id"]}')
    
    return dataset