import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

COMMANDS = [
  'LaneFollow',
  'Left',
  'Straight',
  'Right',
  'ChangeLaneLeft',
  'ChangeLaneRight'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def one_hot_commands(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(categories=[COMMANDS], sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[['command']])
    one_hot_df = pd.DataFrame(data=one_hot_encoded, columns=encoder.get_feature_names_out(['command']))
    return pd.concat([df, one_hot_df], axis=1).drop(columns=['command'])

def preprocess_states(states_df: pd.DataFrame) -> pd.DataFrame:
  states_df = states_df[[
    'frame',
    'pedal_acceleration',
    'steer',
    'speed',
    'acceleration',
    'rotation_rads',
    'compass_rads',
    'gps_compass_bearing',
    'gps_compass_distance',
    'command'
  ]]
  states_df = one_hot_commands(df=states_df)
  return states_df

def load_states(filepath: str) -> Tuple[pd.DataFrame, List[int]]:
  states_df = pd.read_csv(filepath)
  states_df = preprocess_states(states_df=states_df)

  if not states_df['frame'].is_monotonic_increasing:
      states_df.sort_values(by='frame', inplace=True, ignore_index=True)

  states = states_df.drop(columns=['frame'])
  frame_ids = states_df['frame'].to_list()
  return states, frame_ids

def normalize_states(states_df: pd.DataFrame) -> pd.DataFrame:
  states_df['speed'] /= 30.0
  states_df['acceleration'] /= 10.0
  states_df['rotation_rads'] /= 2.0
  states_df['compass_rads'] /= 6.283
  states_df['gps_compass_bearing'] /= np.pi

  # TODO: min max normalization for steer and pedal_acceleration
  return states_df
