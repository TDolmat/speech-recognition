import argparse
import csv
import os
import random
from pydub import AudioSegment
from text_processing import TextProcessing


def main(args):
    data_labels = ['file', 'text']
    data = []
    directory = args.file_path.rpartition('/')[0]
    test_percent = args.test_percent
    train_percent = 100 - test_percent
    convert = args.convert

    # Files loaded from CSV file

    with open(args.file_path, "r") as csv_file:
        audio_files_count = len(csv_file.readlines()) - 1
        print(f"\nFiles found: {audio_files_count}")

    # Iterating through CSV file converting mp3 to wav and creating CSVs

    with open(args.file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            file_name_mp3 = row['path']
            file_name_wav = file_name_mp3.split('.')[0] + ".wav"

            text = TextProcessing.text_with_only_allowed_characters(row['sentence'])
            
            # If there is no clips-wav directory create it
            wav_desctination_directory = directory + "/clips-wav"
            if not os.path.exists(wav_desctination_directory):
                os.makedirs(wav_desctination_directory)

            src = directory + "/clips/" + file_name_mp3
            dst = wav_desctination_directory + "/" + file_name_wav

            # Converting mp3 to wav 
            if convert:
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")

            # Adding converted file path to CSV data
            data.append([dst, text])
            
            # NOTE: Maybe make a progress bar
            print(f"Converted .wav files: ({i}/{audio_files_count})", end="\r")
    data_count = len(data)
    if convert:
        print(f"{data_count} files successfully converted from .mp3 to .wav")

    # Spliting data into train dataset and test dataset

    random.shuffle(data)

    test_data_rows = int(data_count * test_percent / 100)
    train_data_rows = data_count - test_data_rows

    if args.save_csv_path:
        save_csv_path = args.save_csv_path
    else:
        # If save_csv_path is not specified save to directory called datasets-csv
        save_csv_path = directory + "/datasets-csv"
        if not os.path.exists(save_csv_path):
                os.makedirs(save_csv_path)

    print("Creating CSV files")
    with open(save_csv_path + "/train.csv", "w") as train_file:
        train_writer = csv.writer(train_file, delimiter='\t')
        train_writer.writerow(data_labels)
        train_writer.writerows(data[0:train_data_rows])

    with open(save_csv_path + "/test.csv", "w") as test_file:
        test_writer = csv.writer(test_file, delimiter='\t')
        test_writer.writerow(data_labels)
        test_writer.writerows(data[train_data_rows:data_count])
    
    print("\nDone!")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert commonvoice .mp3 files into .wav and create training and testing datasets as CSV files.
    """)

    parser.add_argument('--file_path', type=str, default=None, required=True,
                        help='path to .tsv file from cv-corpus folder')
    parser.add_argument('--save_csv_path', type=str, default=None, required=False,
                        help='path to the dir where the csv files are supposed to be saved')
    parser.add_argument('--test_percent', type=int, default=30, required=False,
                        help='percent of clips for testing (going to test.csv instead of train.csv)')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')

    args = parser.parse_args()
    
    main(args)