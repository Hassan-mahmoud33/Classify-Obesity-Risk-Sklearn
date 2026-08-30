from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib as jb
from pathlib import Path

#---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
FRONDEND_DIR = Path(__file__).resolve().parent.parent / 'front_end'

MODEL_PATH = MODELS_DIR / 'XGB_Classifier_model.pkl'
FEATURES_ENCODER_PATH = MODELS_DIR / "features_encoder.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"


#---------------------------------------------------------------------------

model = jb.load( MODEL_PATH )
features_encoders = jb.load( FEATURES_ENCODER_PATH )
target_encoder = jb.load( TARGET_ENCODER_PATH )

#---------------------------------------------------------------------------

app = FastAPI(title='< Obesity Risk Prediction API >')
app.add_middleware( CORSMiddleware , allow_headers=["*"] )

#---------------------------------------------------------------------------

class PatientData(BaseModel):
    Gender: str                            
    Age: float
    Height: float                          
    Weight: float                            
    family_history_with_overweight: str      
    FAVC: str                            
    FCVC: float
    NCP: float
    CAEC: str                             
    SMOKE: str                           
    CH2O: float
    SCC: str                                 
    FAF: float
    TUE: float
    CALC: str                             
    MTRANS: str                      


#======================================================================================================

def build_features( data : PatientData ) -> pd.DataFrame :

    df = pd.DataFrame( [data.dict()])

    # add all features that you extracted them
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    df['Ideal_Weight'] = (df['Height'] - 1.5) * 50 + 50
    df['Weight_Deviation'] = df['Weight'] - df['Ideal_Weight']
    df['FCVC_NCP_Ratio'] = df['FCVC'] / (df['NCP'] + 1e-6)
    df['FAF_TUE_Ratio'] = df['FAF'] / (df['TUE'] + 1e-6)

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 25, 35, 50, 100],
        labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
                                    )
    
    active_transport = ['Walking', 'Bike']
    df['Active_Transport'] = df['MTRANS'].isin(active_transport).astype(int)


    for col , encoder in features_encoders.items():

        if col in df.columns :

            try :
                df[col] = encoder.transform( df[col].astype(str) )

            except ValueError as e :

                raise HTTPException(status_code= 400 , 
                    detail = f"Unknown value in column '{col}': {df[col].values[0]} ({e})"
                                   )

    df = df[model.feature_names_in_]
    return df


#======================================================================================================


@app.post('/predict')
def predict( data : PatientData ) :

    features = build_features( data )
    pred = model.predict( features )

    pred_label = target_encoder.inverse_transform( pred )[0]

    proba = model.predict_proba( features )[0] 

    confidence = float( proba.max() ) 

    return {'prediction' : pred_label ,
            'confidence' : round( confidence , 2 )
            }


#----------------------------------------------------------------------------------

app.mount("/" , StaticFiles( directory= FRONDEND_DIR , html= True ) , name= 'frond_end')

