# Import helpful libraries
from shutil import move
import os
from pandas import read_csv

"""
Read file location
Go to file location
go file location
copy/move file to new location depending on train, test or val
"""


def split_dataset(data, data_mode: str):
    data_dict = dict()  # dictionary for storing the dataframes
    # Train - 70%
    train = int(0.7 * data.__len__())
    # Test - 15%
    test = int(0.15 * data.__len__())
    # Validation - 15%
    extra = 1 if data_mode == 'age' else 2  # an extra number is added because of error in float computation
    val = int(0.15 * data.__len__()) + extra

    # for each data, it starts counting from the end, then resets the index and drops unnecessary columns
    # noinspection DuplicatedCode
    data_dict['test'] = data[-test:].reset_index().drop(columns=['index'])
    data_dict['val'] = data[-(test + val):-test].reset_index().drop(columns=['index'])
    data_dict['train'] = data[:train].reset_index().drop(columns=['index'])

    # let's check if the partition was done correctly
    if (data_dict['test'].__len__() + data_dict['val'].__len__() + data_dict['train'].__len__()) == data.__len__():
        print("Data was partitioned accurately")
        return data_dict  # returns dictionary with train, test, validation dataframes
    else:
        raise ValueError("Data was NOT partitioned accurately")


# for every file loc, copy file to new location
# set new file location
def move_files(data: dict, data_mode: str):
    dataset_dir = "D:\The Great Big World of Machine Learning" \
                  "\Projects\datasets\AdienceBenchmarkGenderAndAgeClassification\\new_dataset\\"

    for key in data.keys():
        file_list = "\\coarse_tilt_aligned_face." \
                    + data[key]['face_id'].astype(str) \
                    + '.' + data[key]['original_image']
        file_list = list(file_list)

        for count, image in enumerate(file_list):
            source = dataset_dir + data_mode + '\\' + key + '\\' + key + '\\' + image
            target = dataset_dir + data_mode + '\\' + key + '\\' + data[key][data_mode][count] + '\\' + image
            os.makedirs(os.path.dirname(target), exist_ok=True)  # create target folder if it doesn't exist
            dest = move(source, target)

            # code to show verbose
            completion_ratio = (count + 1) / file_list.__len__()
            completion_percentage = completion_ratio * 100
            print(f'loading: {round(completion_percentage, 2)}% - copied {data_mode} {key}', end='\r')
        print()  # clear carriage


def main():
    # Read files
    age_data = read_csv("age_data.csv")
    age_dict = split_dataset(age_data, 'age')
    move_files(age_dict, 'age')

    gender_data = read_csv("gender_data.csv")
    gender_dict = split_dataset(gender_data, 'gender')
    move_files(gender_dict, 'gender')


if __name__ == "__main__":
    main()
