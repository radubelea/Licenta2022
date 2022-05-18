import collections
import csv
import ast
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

champ_labels = {}
crt_label = 0


def get_champ_label(champion):
    global champ_labels
    global crt_label
    if champion not in champ_labels.keys():
        crt_label += 1
        champ_labels[champion] = crt_label
    return champ_labels[champion]


def filter_challenger_data():
    with open('TFT_Challenger_MatchData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('TFT_FilteredData.csv', mode='w') as filtered_file:
            csv_writer = csv.writer(filtered_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0
            for row in csv_reader:
                write_row = []
                if line_count > 0:
                    champ_nr = 0
                    champions = ast.literal_eval(row[7])
                    for key, value in champions.items():
                        write_row.append(get_champ_label(key))
                        for key2, value2 in value.items():
                            if type(value2) is list:
                                item_nr = 0
                                for item in value2:
                                    if item_nr < 3:
                                        write_row.append(item)
                                        item_nr += 1
                                for _ in range(3 - len(value2)):
                                    write_row.append(0)
                            else:
                                write_row.append(value2)
                        champ_nr += 1
                    for _ in range(champ_nr, 12):
                        for _ in range(5):
                            write_row.append(0)
                    write_row.append(row[4])
                    csv_writer.writerow(write_row)
                line_count += 1
        print(f'Processed {line_count} lines.')


def filter_grandmaster_data():
    with open('TFT_Grandmaster_MatchData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('TFT_FilteredData.csv', mode='a') as filtered_file:
            csv_writer = csv.writer(filtered_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0
            for row in csv_reader:
                write_row = []
                if line_count > 0:
                    champ_nr = 0
                    champions = ast.literal_eval(row[7])
                    for key, value in champions.items():
                        write_row.append(get_champ_label(key))
                        for key2, value2 in value.items():
                            if type(value2) is list:
                                item_nr = 0
                                for item in value2:
                                    if item_nr < 3:
                                        write_row.append(item)
                                        item_nr += 1
                                for _ in range(3 - len(value2)):
                                    write_row.append(0)
                            else:
                                write_row.append(value2)
                        champ_nr += 1
                    for _ in range(champ_nr, 12):
                        for _ in range(5):
                            write_row.append(0)
                    write_row.append(row[4])
                    csv_writer.writerow(write_row)
                line_count += 1
        print(f'Processed {line_count} lines.')


def filter_master_data():
    with open('TFT_Master_MatchData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('TFT_FilteredData.csv', mode='a') as filtered_file:
            csv_writer = csv.writer(filtered_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0
            for row in csv_reader:
                write_row = []
                if line_count > 0:
                    champ_nr = 0
                    champions = ast.literal_eval(row[7])
                    for key, value in champions.items():
                        write_row.append(get_champ_label(key))
                        for key2, value2 in value.items():
                            if type(value2) is list:
                                item_nr = 0
                                for item in value2:
                                    if item_nr < 3:
                                        write_row.append(item)
                                        item_nr += 1
                                for _ in range(3 - len(value2)):
                                    write_row.append(0)
                            else:
                                write_row.append(value2)
                        champ_nr += 1
                    for _ in range(champ_nr, 12):
                        for _ in range(5):
                            write_row.append(0)
                    write_row.append(row[4])
                    csv_writer.writerow(write_row)
                line_count += 1
        print(f'Processed {line_count} lines.')


def filter_diamond_data():
    with open('TFT_Diamond_MatchData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('TFT_FilteredData.csv', mode='a') as filtered_file:
            csv_writer = csv.writer(filtered_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0
            for row in csv_reader:
                write_row = []
                if line_count > 0:
                    champ_nr = 0
                    champions = ast.literal_eval(row[7])
                    for key, value in champions.items():
                        write_row.append(get_champ_label(key))
                        for key2, value2 in value.items():
                            if type(value2) is list:
                                item_nr = 0
                                for item in value2:
                                    if item_nr < 3:
                                        write_row.append(item)
                                        item_nr += 1
                                for _ in range(3 - len(value2)):
                                    write_row.append(0)
                            else:
                                write_row.append(value2)
                        champ_nr += 1
                    for _ in range(champ_nr, 12):
                        for _ in range(5):
                            write_row.append(0)
                    write_row.append(row[4])
                    csv_writer.writerow(write_row)
                line_count += 1
        print(f'Processed {line_count} lines.')


def filter_platinum_data():
    with open('TFT_Platinum_MatchData.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with open('TFT_FilteredData.csv', mode='a') as filtered_file:
            csv_writer = csv.writer(filtered_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0
            for row in csv_reader:
                write_row = []
                if line_count > 0:
                    champ_nr = 0
                    champions = ast.literal_eval(row[7])
                    for key, value in champions.items():
                        write_row.append(get_champ_label(key))
                        for key2, value2 in value.items():
                            if type(value2) is list:
                                item_nr = 0
                                for item in value2:
                                    if item_nr < 3:
                                        write_row.append(item)
                                        item_nr += 1
                                for _ in range(3 - len(value2)):
                                    write_row.append(0)
                            else:
                                write_row.append(value2)
                        champ_nr += 1
                    for _ in range(champ_nr, 12):
                        for _ in range(5):
                            write_row.append(0)
                    write_row.append(row[4])
                    csv_writer.writerow(write_row)
                line_count += 1
        print(f'Processed {line_count} lines.')


def retrieve_data():
    dataset = pd.read_csv(r'D:\UBB\an3\sem2\Licenta\ML\TFT_FilteredData.csv')
    print("Obtained dataset")
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print(y)
    print(collections.Counter(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    return x_train, x_test, y_train, y_test


def get_shape(dataset):
    print(dataset.shape)


if __name__ == "__main__":
    x_train,x_test,y_train,y_test = retrieve_data()
    get_shape(x_train)
    get_shape(x_test)
    get_shape(y_train)
    get_shape(y_test)

