import json
from typing import Union

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model/target_action_pipe.pkl')


class Prediction(BaseModel):
    result: float


class Form(BaseModel):
    utm_source: Union[str, None]
    utm_medium: Union[str, None]
    utm_campaign: Union[str, None]
    utm_adcontent: Union[str, None]
    utm_keyword: Union[str, None]
    device_category: Union[str, None]
    device_os: Union[str, None]
    device_brand: Union[str, None]
    device_model: Union[str, None]
    device_screen_resolution: Union[str, None]
    device_browser: Union[str, None]
    geo_country: Union[str, None]
    geo_city: Union[str, None]


@app.get('/status')
def status():
    return 'ok'


@app.get('/version')
def version():
    return model['meta']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model['model'].predict(df)
    return {
        'result' : y[0]
    }

def main():
    data_sessions = pd.read_csv('model/data/ga_sessions.csv', low_memory=False)
    data_sessions_utm_device_geo = data_sessions.iloc[
                                   :,
                                   lambda data_sessions:
                                   data_sessions.columns.str.contains('utm|device|geo', case=False)]
    input_data = data_sessions_utm_device_geo.iloc[[1111]]
    input_data_json = input_data.to_json(orient='records')
    parsed_input_data_json = json.loads(input_data_json)
    with open('model/data/template_form.json', 'w') as file:
        json.dump(parsed_input_data_json[0], file)
    print(input_data)
    y = model['model'].predict(input_data)
    print(f'result: {y[0]}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

