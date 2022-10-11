
import argparse
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import sys
import logging
import os
import glob

import sagemaker


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Parse argument variables passed via the CreateDataset processing step
def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="/opt/ml/processing")
    args, _ = parser.parse_known_args()
    return args


def enrich_data(df_tracks: pd.DataFrame, df_ratings: pd.DataFrame):
    # Perform one-hot encoding
    tracks_tmp = pd.get_dummies(df_tracks, columns=["genre"], prefix="genre")

    # calculate danceability
    tracks_tmp["danceability"] = 0.3*tracks_tmp.valence + 0.1*tracks_tmp.liveness + 0.1*tracks_tmp.energy

    # Join the two dataframes
    tracks_rating = pd.merge(tracks_tmp, df_ratings, how='inner', on='trackId')
    
    num_feat_cols = ['userId', 'energy', 'acousticness', 'valence', 'speechiness', 'instrumentalness', 'liveness', 'tempo', 'danceability', 'genre_Latin', 'genre_Folk',  'genre_Blues', 'genre_Rap', 'genre_Reggae', 'genre_Jazz', 'genre_RnB', 'genre_Country', 'genre_Electronic', 'genre_Pop_Rock']
    
    df_tmp = tracks_rating[tracks_rating.Rating==5][num_feat_cols]
    fivestar_ratings = df_tmp.groupby('userId').mean().add_suffix('_5star').reset_index()

    # Drop columns
    col_drop = ["ratingEventId", "ts", "sessionId", "itemInSession", "trackId"]
    tracks_rating = tracks_rating.drop(col_drop, axis=1)
    
    # join five start rating df with tracks_rating dataframe
    df_output = pd.merge(tracks_rating, fivestar_ratings, how='inner', on='userId').drop(['userId'], axis=1)
    first_col = df_output.pop('Rating')
    df_output.insert(0, 'Rating', first_col)
    df_output.drop_duplicates(inplace=True)
    
    return df_output

def load_data(file_list: list):
    # Define columns to use
    use_cols = []
    # Concat input files
    dfs = []
    for file in file_list:
        if len(use_cols)==0:
            dfs.append(pd.read_csv(file))
        else:
            dfs.append(pd.read_csv(file, usecols=use_cols))    
    return pd.concat(dfs, ignore_index=True)

def save_files(base_dir: str, df_processed: pd.DataFrame):
    
    # split data 
    train, val = train_test_split(df_processed, test_size=0.2, random_state=42)
    val, test = train_test_split(val, test_size=0.05, random_state=42)
    logger.info("Training dataset shape: {}\nValidation dataset shape: {}\nTest dataset shape: {}\n".format(train.shape, val.shape, test.shape))

    # Write train, test splits to output path
    train_output_path = pathlib.Path(f'{base_dir}/output/train')
    val_output_path = pathlib.Path(f'{base_dir}/output/val')
    test_output_path = pathlib.Path(f'{base_dir}/output/test')
    train.to_csv(train_output_path / 'train.csv', header=False, index=False)
    val.to_csv(val_output_path / 'validation.csv', header=False, index=False)
    test.to_csv(test_output_path / 'test.csv', header=False, index=False)

    logger.info('Training, validation, and Testing Sets Created')
    
    return


def main(base_dir: str, args: argparse.Namespace):
    # Input tracks files
    input_dir = os.path.join(base_dir, "input/tracks")
    track_file_list = glob.glob(f"{input_dir}/*.csv")
    logger.info(f"Input file list: {track_file_list}")
        
    if len(track_file_list) == 0:
        raise Exception(f"No input files found in {input_dir}")

    # Input ratings file
    ratings_dir = os.path.join(base_dir, "input/ratings")
    ratings_file_list = glob.glob(f"{ratings_dir}/*.csv")
    logger.info(f"Input file list: {ratings_file_list}")
    if not os.path.exists(ratings_dir):
        raise Exception(f"ratings file does not exist")

    # load data into dataframes
    df_tracks = load_data(track_file_list)
    df_ratings = load_data(ratings_file_list)
    
    # Extract and load taxi zones geopandas dataframe
    df_processed = enrich_data(df_tracks, df_ratings)
    
    return save_files(base_dir, df_processed)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    args = parse_args()
    base_dir = args.base_dir
    main(base_dir, args)
    logger.info("Done")

