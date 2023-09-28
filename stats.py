import datetime
import statistics
import os


def getSeconds(current_time):
    return current_time.hour * 3600 + current_time.minute * 60 + current_time.second

def run():
    people_list = []
    with open("people.txt", 'r') as people_file:
        people_list = [line.rstrip('\n') for line in people_file]

    attendance = {item: ["0", "0", "0", "0", "0", "0", "0", "0"] for item in people_list}

    with open("output/output.csv", 'r') as csv_file:
        for person in people_list:
            flag = False
            total_time = 0
            previous_time = 0
            attendance_counter = 0
            attendance_times = []
            current_date = None
            csv_file.seek(0)
            for line in csv_file:
                current_date = line.split(',')[0]
                if(line.rstrip('\n').endswith(person)):
                    attendance_counter += 1
                    current_time = datetime.datetime.strptime(line.split(',')[2], "%H:%M:%S")
                    current_time_seconds = getSeconds(current_time)
                    if not flag:
                        total_time = current_time_seconds
                        previous_time = current_time_seconds
                        flag = True
                        continue
                    if total_time is not None:
                        total_time = current_time_seconds - total_time
                        if attendance_counter % 2 == 0:
                            attendance_time = current_time_seconds - previous_time
                            attendance_times.append(attendance_time)
                        previous_time = current_time_seconds

            attendance[person][0] = str(current_date)
            if total_time != 0:
                attendance[person][1] = str(datetime.timedelta(seconds=total_time))
                average_time_seconds = total_time // (attendance_counter // 2)
                attendance[person][2] = str(datetime.timedelta(seconds=average_time_seconds))
                min_time = min(attendance_times)
                max_time = max(attendance_times)
                median_time = statistics.median(attendance_times)
                attendance[person][3] = str(datetime.timedelta(seconds=min_time))
                attendance[person][4] = str(datetime.timedelta(seconds=max_time))
                attendance[person][5] = str(datetime.timedelta(seconds=median_time))
                if len(attendance_times) > 1:
                    standard_dev = statistics.stdev(attendance_times)
                    attendance[person][6] = str(standard_dev)
 
    for key, value in attendance.items():
        if value[0] == "":
            print(key, "nije prisutan/na.")
        else:
            print(key, "Datum:", value[0], "Ukupno:", value[1], "Prosjecno:", value[2], "Min:", value[3], "Max:", value[4], "Median:", value[5],
                  "Stand dev:", value[6])

    with open("output/report.csv", 'a+') as report_file:
        if os.path.getsize("output/report.csv") == 0:
            report_file.write("date,name,total_time,avg_time,min_time,max_time,median_time,stdev,absence\n")
        for key, value in attendance.items():
            absence = 0
            if value[1] == "0":
                today = datetime.datetime.strptime(value[0], '%Y-%m-%d').date()
                yesterday = str(today - datetime.timedelta(days=1))
                report_file.seek(0)
                for line in report_file:
                    split_line = line.split(',')
                    if (split_line[0] == yesterday) & (split_line[1] == key):
                        absence = int(split_line[8]) + 1

            report_file.write(value[0] + "," + str(key) + "," + value[1] + "," + value[2] + "," + value[3] + "," + value[4] + "," + value[5] + "," + value[6] + "," + str(absence) + "\n")