import csv

# Open the CSV file
state=0
sample_values={}
with open('D:\\Projects\\GraphST\\result\\photo.csv', mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate through the rows of the file
    state = 0
    current=''
    # 5 6 7 10 11
    serial=7
    for row in csv_reader:
        print(row)
        if row[0][0]=='1' and state==1:

            if row[0] not in sample_values.keys():
                sample_values[row[0]]=(row[1],row[serial])

            else:
                if float(sample_values[row[0]][1])<float(row[serial]):
                    # print("SMALL:"+ str(type(sample_values[row[0]][1])))
                    # print("BIG:"+str(type(row[serial])))
                    sample_values[row[0]]=(row[1],row[serial])

        if row[0][0]!='1':
            print(row[0])
            if  row[0] in ['New','New+Scatter','Scatter','Enco+Neighbour','Senco','Feat+Point']:
                print("OK")
                state=1
            else:
                state=0
                #current=row[0]




        # Print each row
print(sample_values)
for i in sample_values.keys():
    print(sample_values[i][0])
