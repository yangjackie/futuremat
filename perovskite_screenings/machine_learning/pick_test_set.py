import random

from perovskite_screenings.halide_double_perovskites.analysis import *

dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')

all_uids = []
_db = sqlite3.connect(dbname)
cur = _db.cursor()
cur.execute("SELECT * FROM systems")
rows = cur.fetchall()

for row in rows:
    for i in row:
        if 'uid' in str(i):
            this_dict = json.loads(str(i))
            this_uid = this_dict['uid']
            if 'dpv' in this_uid:
                all_uids.append(this_uid)

db = connect(dbname)

sigma_dict = {}

for uid in all_uids:
    row = None
    sigma = None

    try:
        row = db.get(selection=[('uid', '=', uid)])
    except:
        pass
    if row is not None:
        try:
            sigma = row.key_value_pairs['sigma_300K_single']
        except KeyError:
            pass

    if sigma is not None:
        sigma_dict[uid]=sigma

#=====================================Batch I=============================================
band = {}
random_entries = {}
for key in sigma_dict.keys():
    if sigma_dict[key]<=0.5:
        band[key] = sigma_dict[key]
entry_list = list(band.items())
while len(random_entries) < 50:
    random_entry = random.choice(entry_list)
    if random_entry not in random_entries:
        random_entries[random_entry[0]] = random_entry[1]

entries = {k: v for k, v in sorted(random_entries.items(), key=lambda item: item[1])}
for k in entries.keys():
    print(k,"{:.3f}".format(entries[k]))


# =====================================Batch II=============================================
band = {}
random_entries = {}
for key in sigma_dict.keys():
    if (sigma_dict[key] > 0.5) and  (sigma_dict[key] <= 0.75):
        band[key] = sigma_dict[key]
entry_list = list(band.items())
while len(random_entries) < 25:
    random_entry = random.choice(entry_list)
    if random_entry not in random_entries:
        random_entries[random_entry[0]] = random_entry[1]

entries = {k: v for k, v in sorted(random_entries.items(), key=lambda item: item[1])}
for k in entries.keys():
    print(k, "{:.3f}".format(entries[k]))


# =====================================Batch III=============================================
band = {}
random_entries = {}
for key in sigma_dict.keys():
    if (sigma_dict[key] > 0.75) and  (sigma_dict[key] <= 1):
        band[key] = sigma_dict[key]
entry_list = list(band.items())
while len(random_entries) < 25:
    random_entry = random.choice(entry_list)
    if random_entry not in random_entries:
        random_entries[random_entry[0]] = random_entry[1]

entries = {k: v for k, v in sorted(random_entries.items(), key=lambda item: item[1])}
for k in entries.keys():
    print(k, "{:.3f}".format(entries[k]))

# =====================================Batch IV=============================================
band = {}
random_entries = {}
for key in sigma_dict.keys():
    if (sigma_dict[key] > 1) and  (sigma_dict[key] <= 2.5):
        band[key] = sigma_dict[key]
entry_list = list(band.items())
while len(random_entries) < 25:
    random_entry = random.choice(entry_list)
    if random_entry not in random_entries:
        random_entries[random_entry[0]] = random_entry[1]

entries = {k: v for k, v in sorted(random_entries.items(), key=lambda item: item[1])}
for k in entries.keys():
    print(k, "{:.3f}".format(entries[k]))