import csv


def read_csv(path):
    rows = {}
    with open(path, newline='') as infile:
        reader = csv.reader(infile, delimiter='\t', quotechar='|')
        next(reader, None)  # skip headers
        for row in reader:
            # remove 'id1' from e.g. 'id10001', then convert '0001' to 1
            rows[int(row[0][3:])] = row[1]
    return rows
    

if __name__ == '__main__':
    rows = read_csv('vox1_meta.csv')
    print(rows[1])
