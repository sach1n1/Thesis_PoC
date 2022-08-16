from datetime import datetime

Value1 = open('../data/Value1_bk_raw.csv', 'w')
Value2 = open('../data/Value2_bk_raw.csv', 'w')

with open('../data/Value.csv', 'r') as value:
    for lines in value:
        linesp = lines.split(',')
        try:
            if linesp[1] == "1":
                #Value1.write(datetime.fromtimestamp(int(linesp[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S') + ',' +
                #             linesp[2])
                Value1.write((linesp[0]) + ',' +linesp[2])
            if linesp[1] == "2":
                #Value2.write(datetime.fromtimestamp(int(linesp[0]) / 1000).strftime('%Y-%m-%d %H:%M:%S') + ',' +
                #             linesp[2])
                Value2.write((linesp[0]) + ',' + linesp[2])
        except ValueError:
            pass

value.close()
Value1.close()
Value2.close()