import csv
import platform
from korea_news_crawler.exceptions import *


class Writer(object):
    def __init__(self, category, article_category, date, page_num, total_page):

        self.start_year = date[0:4]
        self.start_month = date[4:6]
        self.start_day = date[6:8]



        self.file = None
        self.initialize_file(category, article_category)

        self.csv_writer = csv.writer(self.file)


    def initialize_file(self, category, article_category):

        output_path = f'../output'
        if os.path.exists(output_path) is not True:
            os.mkdir(output_path)

        if os.path.exists(output_path+"/"+article_category) is not True:
            os.mkdir(output_path+"/"+article_category)

        # print("output_path+article_category >>>> ", output_path+"/"+article_category)

        if os.path.exists(output_path + "/" + article_category + "/" + str(self.start_year)) is not True:
            os.mkdir(output_path+"/"+article_category + "/" + str(self.start_year))

        if os.path.exists(output_path + "/" + article_category + "/" + str(self.start_year) + "/" + str(self.start_month)) is not True:
            os.mkdir(output_path+"/"+article_category + "/" + str(self.start_year) + "/" + str(self.start_month))

        if os.path.exists(output_path + "/" + article_category + "/" + str(self.start_year) + "/" + str(self.start_month) + "/" + str(self.start_day)) is not True:
            os.mkdir(output_path+"/"+article_category + "/" + str(self.start_year) + "/" + str(self.start_month) + "/" + str(self.start_day))

        file_name = f'{output_path}/{article_category}/{self.start_year}/{self.start_month}/{self.start_day}/{category}_{article_category}_{self.start_year}{self.start_month}{self.start_day}.csv'
        # if os.path.isfile(file_name):
        #     raise ExistFile(file_name)

        user_os = str(platform.system())
        if user_os == "Windows":
            self.file = open(file_name, 'a', encoding='euc-kr', newline='')
        # Other OS uses utf-8
        else:
            self.file = open(file_name, 'a', encoding='utf-8', newline='')

    def write_row(self, arg):
        self.csv_writer.writerow(arg)

    def close(self):
        self.file.close()
